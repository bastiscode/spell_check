import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class APIResult {
  int statusCode;
  String message;
  dynamic value;

  APIResult(this.statusCode, this.message, this.value);
}

class API {
  String _baseURL = "";

  API._privateConstructor() {
    if (kIsWeb) {
      _baseURL = "http://${Uri.base.host}:12345";
    } else if (Platform.isAndroid) {
      _baseURL = "http://10.0.2.2:12345";
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
      final res = await http.get(Uri.parse("$_baseURL/models"));
      return jsonDecode(res.body);
    } catch (e) {
      return null;
    }
  }

  dynamic info() async {
    try {
      final res = await http.get(Uri.parse("$_baseURL/info"));
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
              "$_baseURL/process_text?pipeline=${Uri.encodeComponent(pipeline)}&edited=${edited ? "true" : "false"}"),
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
}

final api = API.instance;
