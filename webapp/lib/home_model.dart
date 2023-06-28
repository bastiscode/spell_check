import 'dart:async';
import 'dart:collection';
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:webapp/api.dart';
import 'package:webapp/base_model.dart';
import 'package:webapp/components/message.dart';

class HomeModel extends BaseModel {
  dynamic _models;

  dynamic get models => _models;

  dynamic _examples;

  dynamic get examples => _examples;

  dynamic get available => _models != null;

  dynamic _info;

  dynamic get info => _info;

  static const maxLiveLength = 512;

  String? trModel;
  String? sedwModel;
  String? secModel;
  dynamic outputs;
  dynamic runtimes;
  List<String> input = [];

  // sedw options (e.g. threshold)
  // TODO: implement these

  // sec options (e.g. greedy or beam search)
  // TODO: implement these

  String? lastInputString;
  late TextEditingController inputController;
  late TextEditingController outputController;

  bool get validPipeline =>
      trModel != null || sedwModel != null || secModel != null;

  List<dynamic> getModels(String task) {
    return _models.where((element) => element["task"] == task).toList();
  }

  dynamic getModel(String task, String name) {
    return _models.firstWhere(
        (element) => element["task"] == task && element["name"] == name);
  }

  bool _ready = false;

  bool get ready => _ready;

  bool _waiting = false;

  bool get waiting => _waiting;

  bool _hasResults = false;

  bool get hasResults => _hasResults;

  bool get hasInput => inputController.text.isNotEmpty;

  bool live = false;

  bool hidePipeline = false;

  String truncate(String text) {
    return live ? text.substring(0, min(maxLiveLength, text.length)) : text;
  }

  Future<void> init(TextEditingController inputController,
      TextEditingController outputController) async {
    this.inputController = inputController;
    this.outputController = outputController;

    _models = await api.models();
    final prefs = await SharedPreferences.getInstance();
    final hidePipeline = prefs.getBool("hidePipeline");
    if (hidePipeline != null) {
      this.hidePipeline = hidePipeline;
    }
    final List<String>? pipeline = prefs.getStringList("pipeline");
    if (available && pipeline != null) {
      if (pipeline[0] != "" &&
          getModels("whitespace correction").firstWhere(
                  (element) => element["name"] == pipeline[0],
                  orElse: () => null) !=
              null) {
        trModel = pipeline[0];
      }
      if (pipeline[1] != "" &&
          getModels("sed words").firstWhere(
                  (element) => element["name"] == pipeline[1],
                  orElse: () => null) !=
              null) {
        sedwModel = pipeline[1];
      }
      if (pipeline[2] != "" &&
          getModels("sec").firstWhere(
                  (element) => element["name"] == pipeline[2],
                  orElse: () => null) !=
              null) {
        secModel = pipeline[2];
      }
    }
    _info = await api.info();
    _examples = await api.examples();
    _ready = true;
    notifyListeners();
  }

  savePipeline() async {
    final prefs = await SharedPreferences.getInstance();
    prefs.setStringList(
      "pipeline",
      <String>[trModel ?? "", sedwModel ?? "", secModel ?? ""],
    );
    prefs.setBool("hidePipeline", hidePipeline);
  }

  runPipelineLive() async {
    while (live) {
      final inputString = inputController.text;
      if (lastInputString != inputString &&
          !waiting &&
          validPipeline &&
          inputController.text.length <= maxLiveLength) {
        final error = await runPipeline(inputString);
        if (error == null) {
          lastInputString = inputString;
        }
      } else {
        // wait for some time
        await Future.delayed(const Duration(milliseconds: 10), () => {});
      }
    }
  }

  Future<Message?> runPipeline(String inputString) async {
    _waiting = true;
    if (!live) {
      _hasResults = false;
      outputs = null;
      runtimes = null;
      input.clear();
    }
    notifyListeners();
    final inputLines = inputString
        .split("\n")
        .map((s) => s.trim().replaceAll(RegExp(r"\s+"), " "))
        .toList();
    var inputText = "${inputLines.join("\n")}\n";
    final result =
        await api.runPipeline(inputText, trModel, sedwModel, secModel);
    final error = errorMessageFromAPIResult(result, "error running pipeline");
    if (error == null) {
      outputs = result.value["output"];
      runtimes = result.value["runtimes"];
      input = inputLines;
      if (secModel != null) {
        outputController.text = outputs["sec"]["text"].join("\n");
      } else if (sedwModel != null) {
        outputController.text = outputs["sed words"]["detections"]
            .map((detection) => detection.join(" "))
            .join("\n");
      } else {
        outputController.text =
            outputs["whitespace correction"]["text"].join("\n");
      }
    }
    _hasResults = error == null;
    _waiting = false;
    notifyListeners();
    return error;
  }

  Future<APIResult> evaluateTr(String groundtruth) async {
    return await api.evaluateTr(input.join("\n"),
        outputs["whitespace correction"]["text"].join("\n"), groundtruth);
  }

  Future<APIResult> evaluateSedw(String groundtruth) async {
    String inputString;
    if (outputs.containsKey("whitespace correction")) {
      inputString = outputs["whitespace correction"]["text"].join("\n");
    } else {
      inputString = input.join("\n");
    }
    List<String> rawDetections = [];
    for (final detection in outputs["sed words"]["detections"]) {
      rawDetections.add(detection.join(" "));
    }
    return await api.evaluateSedw(
        inputString, rawDetections.join("\n"), groundtruth);
  }

  Future<APIResult> evaluateSec(String groundtruth) async {
    String inputString;
    if (outputs.containsKey("sed words")) {
      inputString = outputs["sed words"]["text"].join("\n");
    } else if (outputs.containsKey("whitespace correction")) {
      inputString = outputs["whitespace correction"]["text"].join("\n");
    } else {
      inputString = input.join("\n");
    }
    return await api.evaluateSec(
        inputString, outputs["sec"]["text"].join("\n"), groundtruth);
  }
}
