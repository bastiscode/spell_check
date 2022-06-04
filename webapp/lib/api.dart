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

  Future<APIResult> repair(String text, String model,
      [bool edited = true]) async {
    try {
      final res = await http.post(
          Uri.parse(
              "$_baseURL/process_text?task=${Uri.encodeComponent("tokenization repair")}&model=${Uri.encodeComponent(model)}&edited=${edited ? "true" : "false"}"),
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

  Future<APIResult> detect(String text, String model) async {
    try {
      final res = await http.post(
          Uri.parse(
              "$_baseURL/process_text?task=${Uri.encodeComponent("sed words")}&model=${Uri.encodeComponent(model)}"),
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

  Future<APIResult> correct(String text, String model,
      [bool edited = true, String? detections]) async {
    try {
      final res = await http.post(
          Uri.parse(
              "$_baseURL/process_text?task=sec&model=${Uri.encodeComponent(model)}&edited=${edited ? "true" : "false"}"),
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
