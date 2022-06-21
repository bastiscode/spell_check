import 'dart:async';
import 'dart:collection';

import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:webapp/api.dart';
import 'package:webapp/base_model.dart';

enum Status { error, warn, info }

class Message {
  String message;
  Status status;

  Message(this.message, this.status);
}

class HomeModel extends BaseModel {
  dynamic _models;

  dynamic get available => _models != null;

  dynamic _info;

  dynamic get info => _info;

  String? trModel;
  String? sedwModel;
  String? secModel;
  dynamic outputs;
  dynamic runtimes;
  List<String> input = [];

  String? fileString;

  String lastInputString = "";
  String inputString = "";

  bool get validPipeline =>
      trModel != null || sedwModel != null || secModel != null;

  List<String> get tasks => _models["default"].keys.toList();

  List<dynamic> getModels(String task) {
    return _models.where((element) => element["task"] == task).toList();
  }

  bool _ready = false;

  bool get ready => _ready;

  bool _waiting = false;

  bool get waiting => _waiting;

  bool _hasResults = false;

  bool get hasResults => _hasResults;

  bool live = false;

  bool hidePipeline = false;

  final Queue<Message> _messages = Queue<Message>();

  Future<void> init() async {
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

  bool get messageAvailable {
    return _messages.isNotEmpty;
  }

  void addMessage(Message message) {
    _messages.add(message);
  }

  Message popMessage() {
    return _messages.removeFirst();
  }

  bool _checkApiResult(APIResult result, String messagePrefix) {
    if (result.statusCode == -1) {
      addMessage(
          Message("$messagePrefix: unable to reach server", Status.error));
      return false;
    } else if (result.statusCode != 200) {
      addMessage(Message("$messagePrefix: ${result.message}", Status.warn));
      return false;
    } else {
      return true;
    }
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
      final inputString = this.inputString;
      if (lastInputString != inputString && !waiting) {
        final success = await runPipeline();
        if (success) {
          lastInputString = inputString;
        }
      } else {
        // wait for some time
        await Future.delayed(const Duration(milliseconds: 10), () => {});
      }
    }
  }

  Future<bool> runPipeline() async {
    _waiting = true;
    if (!live) {
      outputs = null;
      runtimes = null;
    }
    notifyListeners();
    final inputLines = inputString
        .split("\n")
        .map((s) => s.trim().replaceAll(RegExp(r"\s+"), " "))
        .toList();
    var inputText = "${inputLines.join("\n")}\n";
    final result =
        await api.runPipeline(inputText, trModel, sedwModel, secModel);
    final success = _checkApiResult(result, "error running pipeline");
    if (success) {
      outputs = result.value["output"];
      runtimes = result.value["runtimes"];
      input = inputLines;
    }
    _hasResults = success;
    _waiting = false;
    notifyListeners();
    return _hasResults;
  }

  onClipboard(String name) {
    addMessage(
      Message("Copied $name outputs to clipboard", Status.info),
    );
    notifyListeners();
  }

  onDownload(String? fileName, String name) {
    if (fileName != null) {
      addMessage(
        Message("Downloaded $name outputs to $fileName", Status.info),
      );
    } else {
      addMessage(Message("Unexpected error downloading $name outputs", Status.error));
    }
    notifyListeners();
  }
}
