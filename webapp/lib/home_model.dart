import 'dart:async';
import 'dart:collection';

import 'package:shared_preferences/shared_preferences.dart';
import 'package:webapp/api.dart';
import 'package:webapp/base_model.dart';
import 'package:webapp/components/message.dart';


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

  Message? _checkApiResult(APIResult result, String messagePrefix) {
    if (result.statusCode == -1) {
      return Message("$messagePrefix: unable to reach server", Status.error);
    } else if (result.statusCode != 200) {
      return Message("$messagePrefix: ${result.message}", Status.warn);
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
        final error = await runPipeline();
        if (error == null) {
          lastInputString = inputString;
        }
      } else {
        // wait for some time
        await Future.delayed(const Duration(milliseconds: 10), () => {});
      }
    }
  }

  Future<Message?> runPipeline() async {
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
    final error = _checkApiResult(result, "error running pipeline");
    if (error == null) {
      outputs = result.value["output"];
      runtimes = result.value["runtimes"];
      input = inputLines;
    }
    _hasResults = error == null;
    _waiting = false;
    notifyListeners();
    return error;
  }
}
