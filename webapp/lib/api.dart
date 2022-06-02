import 'package:flutter/cupertino.dart';
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
    _baseURL = "http://0.0.0.0:12345";
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

  Future<APIResult> repair(String text, String model) async {
    try {
      final res = await http.post(
          Uri.parse(
              "$_baseURL/process_text?task=${Uri.encodeComponent("tokenization repair")}&model=${Uri.encodeComponent(model)}"),
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
      {String? detections}) async {
    try {
      final res = await http.post(
          Uri.parse(
              "$_baseURL/process_text?task=sec&model=${Uri.encodeComponent(model)}"),
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
