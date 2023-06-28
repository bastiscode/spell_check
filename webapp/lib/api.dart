import  "dart:io";

import 'package:http/http.dart' as http;
import 'dart:convert';

import 'package:webapp/components/message.dart';
import 'package:window_location_href/window_location_href.dart';

class APIResult {
  int statusCode;
  String message;
  dynamic value;

  APIResult(this.statusCode, this.message, this.value);
}

class API {
  String _apiBaseURL = "";
  String _webBaseURL = "";

  API._privateConstructor() {
    final href = getHref();
    if (href != null) {
      _apiBaseURL = href.endsWith("/") ? "${href}api" : "$href/api";
      // for local development
      // _apiBaseURL = "http://0.0.0.0:44444";
      _webBaseURL = href;
    } else if (Platform.isAndroid) {
      // for local development on an android emulator
      _apiBaseURL = "http://10.0.2.2:44444";
      _webBaseURL = "http://10.0.2.2:8080";
    } else {
      throw UnsupportedError("unknown platform");
    }
  }

  static final API _instance = API._privateConstructor();

  static API get instance {
    return _instance;
  }

  dynamic models() async {
    try {
      final res = await http.get(Uri.parse("$_apiBaseURL/models"));
      return jsonDecode(res.body);
    } catch (e) {
      return null;
    }
  }

  dynamic info() async {
    try {
      final res = await http.get(Uri.parse("$_apiBaseURL/info"));
      return jsonDecode(res.body);
    } catch (e) {
      return null;
    }
  }

  dynamic examples() async {
    try {
      final res = await http.get(Uri.parse("$_webBaseURL/assets/examples/examples.json"));
      return jsonDecode(res.body);
    } catch (e) {
      return null;
    }
  }

  Future<APIResult> runPipeline(String text, String? tokenizationRepairModel,
      String? sedWordsModel, String? secModel,
      {bool edited = true}) async {
    try {
      final pipeline =
          "${tokenizationRepairModel ?? ""},${sedWordsModel ?? ""},${secModel ?? ""}";
      final res = await http.post(
          Uri.parse(
              "$_apiBaseURL/run?pipeline=${Uri.encodeComponent(pipeline)}&edited=${edited ? "true" : "false"}"),
          body: {"text": text});
      if (res.statusCode != 200) {
        return APIResult(res.statusCode, res.body, null);
      } else {
        return APIResult(res.statusCode, "ok", jsonDecode(res.body));
      }
    } catch (e) {
      return APIResult(-1, "could not reach backend", null);
    }
  }

  Future<APIResult> evaluateSec(
      String input, String prediction, String groundtruth) async {
    return await _evaluate("sec", input, prediction, groundtruth);
  }

  Future<APIResult> evaluateSedw(
      String input, String prediction, String groundtruth) async {
    return await _evaluate("sed words", input, prediction, groundtruth);
  }

  Future<APIResult> evaluateTr(
      String input, String prediction, String groundtruth) async {
    return await _evaluate(
        "whitespace correction", input, prediction, groundtruth);
  }

  Future<APIResult> _evaluate(
      String task, String input, String prediction, String groundtruth) async {
    try {
      final res = await http.post(
          Uri.parse("$_apiBaseURL/eval?task=${Uri.encodeComponent(task)}"),
          body: {
            "input": input,
            "prediction": prediction,
            "groundtruth": groundtruth
          });
      if (res.statusCode != 200) {
        return APIResult(res.statusCode, res.body, null);
      } else {
        return APIResult(res.statusCode, "ok", jsonDecode(res.body));
      }
    } catch (e) {
      return APIResult(-1, "could not reach backend", null);
    }
  }
}

final api = API.instance;


Message? errorMessageFromAPIResult(APIResult result, String messagePrefix) {
  if (result.statusCode == -1) {
    return Message("$messagePrefix: unable to reach server", Status.error);
  } else if (result.statusCode != 200) {
    return Message("$messagePrefix: ${result.message}", Status.warn);
  }
  return null;
}
