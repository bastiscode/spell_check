import 'dart:collection';

import 'package:flutter/cupertino.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:webapp/api.dart';
import 'package:webapp/base_model.dart';

enum Status { error, warn, info }

class Message {
  String message;
  Status status;

  Message(this.message, this.status);
}

const taskIdentifierToTaskName = {
  "sec": "Spelling error correction",
  "sed words": "Word-level spelling error detection",
  "sed sequence": "Sequence-level spelling error detection",
  "tokenization repair": "Tokenization repair"
};

class HomeModel extends BaseModel {
  dynamic _models;

  dynamic get available => _models != null;

  dynamic _info;

  dynamic get info => _info;

  String? trModel;
  String? sedwModel;
  String? secModel;
  dynamic trResults;
  dynamic sedwResults;
  dynamic secResults;

  String? fileString = "";

  String inputString = "";
  List<String> input = [];

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

  bool live = false;

  Queue<Message> _messages = Queue<Message>();

  Future<void> init() async {
    _models = await api.models();
    final prefs = await SharedPreferences.getInstance();
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
      addMessage(Message("$messagePrefix: result.message", Status.warn));
      return false;
    } else {
      return true;
    }
  }

  void savePipeline() async {
    final prefs = await SharedPreferences.getInstance();
    prefs.setStringList(
      "pipeline",
      <String>[
        trModel != null ? trModel! : "",
        sedwModel != null ? sedwModel! : "",
        secModel != null ? secModel! : ""
      ],
    );
  }

  Future<void> runPipeline() async {
    _waiting = true;
    if (!live) {
      trResults = null;
      sedwResults = null;
      secResults = null;
    }
    notifyListeners();
    final inputLines = inputString
        .split("\n")
        .map((s) => s.trim().replaceAll(RegExp(r"\s+"), " "))
        .toList();
    var inputText = "${inputLines.join("\n")}\n";
    bool pipelineFail = false;
    if (trModel != null) {
      final result = await api.repair(inputText, trModel!);
      final success =
          _checkApiResult(result, "error at tokenization repair step");
      if (success) {
        trResults = result.value;
        inputText = trResults["output"]["text"].join("\n");
      } else {
        pipelineFail = true;
      }
    }
    if (sedwModel != null && !pipelineFail) {
      final result = await api.detect(inputText, sedwModel!);
      final success = _checkApiResult(
          result, "error at word-level spelling error detection step");
      if (success) {
        sedwResults = result.value;
        inputText = sedwResults["output"]["text"].join("\n");
      } else {
        pipelineFail = true;
      }
    }
    if (secModel != null && !pipelineFail) {
      final result = await api.correct(inputText, secModel!);
      final success =
          _checkApiResult(result, "error at spelling error correction step");
      if (success) {
        secResults = result.value;
      }
    }
    input = inputLines;
    _waiting = false;
    notifyListeners();
  }
}
