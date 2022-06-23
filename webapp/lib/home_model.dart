import 'dart:async';
import 'dart:collection';

import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:webapp/api.dart';
import 'package:webapp/base_model.dart';
import 'package:webapp/components/message.dart';


class HomeModel extends BaseModel {
  dynamic _models;

  dynamic get models => _models;

  dynamic get available => _models != null;

  dynamic _info;

  dynamic get info => _info;

  String? trModel;
  String? sedwModel;
  String? secModel;
  dynamic outputs;
  dynamic runtimes;
  List<String> input = [];

  String? trGroundtruth;
  String? sedwGroundtruth;
  String? secGroundtruth;

  dynamic trEvaluation;
  dynamic sedwEvaluation;
  dynamic secEvaluation;

  String? lastInputString;
  late TextEditingController inputController;
  late TextEditingController outputController;

  bool get validPipeline =>
      trModel != null || sedwModel != null || secModel != null;

  List<dynamic> getModels(String task) {
    return _models.where((element) => element["task"] == task).toList();
  }

  dynamic getModel(String task, String name) {
    return _models.firstWhere((element) => element["task"] == task && element["name"] == name);
  }

  bool _ready = false;

  bool get ready => _ready;

  bool _waiting = false;

  bool get waiting => _waiting;

  bool _evaluating = false;

  bool get evaluating => _evaluating;

  bool _hasResults = false;

  bool get hasResults => _hasResults;

  bool get hasInput => inputController.text.isNotEmpty;

  bool live = false;

  bool hidePipeline = false;

  Future<void> init(TextEditingController inputController, TextEditingController outputController) async {
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
          getModels("tokenization repair").firstWhere(
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
      if (lastInputString != inputString && !waiting && validPipeline) {
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
    debugPrint("num input lines: ${inputLines.length}");
    var inputText = inputLines.join("\n");
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
        outputController.text = outputs["sed words"]["detections"].map((detection) => detection.join(" ")).join("\n");
      } else {
        outputController.text = outputs["tokenization repair"]["text"].join("\n");
      }
    }
    _hasResults = error == null;
    _waiting = false;
    notifyListeners();
    return error;
  }

  Future<Message?> evaluateTr() async {
    _evaluating = true;
    notifyListeners();
    final result = await api.evaluateTr(input.join("\n"), outputs["tokenization repair"]["text"].join("\n"), trGroundtruth!);
    final error = errorMessageFromAPIResult(result, "error evaluating tokenization repair");
    if (error == null) {
      trEvaluation = result.value;
    }
    _evaluating = false;
    notifyListeners();
    return error;
  }

  Future<Message?> evaluateSedw() async {
    _evaluating = true;
    notifyListeners();
    String inputString;
    if (outputs.containsKey("tokenization repair")) {
      inputString = outputs["tokenization repair"]["text"].join("\n");
    } else {
      inputString = input.join("\n");
    }
    List<String> rawDetections = [];
    for (final detection in outputs["sed words"]["detections"]) {
      rawDetections.add(detection.join(" "));
    }
    final result = await api.evaluateSedw(inputString, rawDetections.join("\n"), sedwGroundtruth!);
    final error = errorMessageFromAPIResult(result, "error evaluating spelling error detection");
    if (error == null) {
      sedwEvaluation = result.value;
    }
    _evaluating = false;
    notifyListeners();
    return error;
  }

  Future<Message?> evaluateSec() async {
    _evaluating = true;
    notifyListeners();
    String inputString;
    if (outputs.containsKey("sed words")) {
      inputString = outputs["sed words"]["text"].join("\n");
    } else if (outputs.containsKey("tokenization repair")) {
      inputString = outputs["tokenization repair"]["text"].join("\n");
    } else {
      inputString = input.join("\n");
    }
    final result = await api.evaluateSec(inputString, outputs["sec"]["text"].join("\n"), secGroundtruth!);
    final error = errorMessageFromAPIResult(result, "error evaluating spelling error correction");
    if (error == null) {
      secEvaluation = result.value;
    }
    _evaluating = false;
    notifyListeners();
    return error;
  }
}
