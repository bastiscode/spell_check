import 'dart:collection';

import 'package:flutter/material.dart';
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

  String trModel = "";
  String sedwModel = "";
  String secModel = "";

  bool get validPipeline =>
      trModel != "" || sedwModel != "" || secModel != "";

  List<String> get tasks => _models["default"].keys.toList();

  List<dynamic> models(String task) {
    return _models["models"]
        .where((element) => element["task"] == task)
        .toList();
  }

  bool _ready = false;

  bool get ready => _ready;

  bool _waiting = false;

  String resultText = "";

  bool get waiting => _waiting;
  
  bool live = false;

  Queue<Message> _messages = Queue<Message>();

  Future<void> init() async {
    _models = await api.models();
    _ready = true;
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

  bool _checkApiResult(APIResult result) {
    if (result.statusCode == -1) {
      addMessage(Message("unable to reach server", Status.error));
      return false;
    } else if (result.statusCode != 200) {
      addMessage(Message(result.message, Status.warn));
      return false;
    } else {
      return true;
    }
  }

  Future<void> runPipeline(String text) async {
    _waiting = true;
    notifyListeners();
    if (trModel != "") {
      final result = await api.repair(text, trModel);
      final success = _checkApiResult(result);
      if (success) {
        text = result.value["output"]["text"].join("\n");
        resultText = text;
      }
    }
    if (sedwModel != "") {
      final result = await api.detect(text, sedwModel);
      final success = _checkApiResult(result);
      if (success) {
        text = result.value["output"]["text"].join("\n");
        resultText = text;
      }
    }
    if (secModel != "") {
      final result = await api.correct(text, secModel);
      final success = _checkApiResult(result);
      if (success) {
        text = result.value["output"]["text"].join("\n");
        resultText = text;
      }
    }
    _waiting = false;
    notifyListeners();
  }
}
