import 'dart:convert' show utf8, base64;
import "dart:html";
import 'package:intl/intl.dart';
import 'package:webapp/platforms/file_download_interface.dart';

class WebFileDownloader implements FileDownloader {
  @override
  Future<String?> downloadFile(String content, String name) async {
    try {
      final now = DateTime.now();
      final nowFormatted = DateFormat("yyyy_MM_dd_HH_mm_ss").format(now);
      final fileName = "${name}_$nowFormatted.txt";
      final base64Content = base64.encode(utf8.encode(content));
      AnchorElement(
        href:
            "data:application/octet-stream;charset=utf-16le;base64,$base64Content",
      )
        ..setAttribute("download", fileName)
        ..click();
      return fileName;
    } catch (e) {
      return null;
    }
  }
}

FileDownloader getFileDownloader() => WebFileDownloader();
