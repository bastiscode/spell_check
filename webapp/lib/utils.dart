import 'dart:convert';

import 'package:flutter/cupertino.dart';

String formatS(double s) {
  if (s < 1) {
    final ms = s * 1000;
    return "${ms.round()}ms";
  } else {
    return "${s.toStringAsFixed(2)}s";
  }
}

int numBytes(String s) {
  return utf8.encode(s).length;
}

String formatB(double b) {
  if (b > 1000) {
    b /= 1000;
    return "${b.toStringAsFixed(2)}kB";
  } else {
    return "${b.round()}b";
  }
}

helperTextFromTextController(TextEditingController controller) {
  final numInputLines = controller.text.split("\n").length;
  final numInputBytes = numBytes(controller.text);
  return "$numInputLines line${numInputLines > 1 ? "s" : ""} with ${formatB(numInputBytes.toDouble())} of text";
}
