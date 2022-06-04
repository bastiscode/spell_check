import 'dart:io';

import 'package:intl/intl.dart';
import 'package:path_provider/path_provider.dart';
import 'package:webapp/platforms/file_download_interface.dart';

class NativeFileDownloader implements FileDownloader {
  @override
  Future<String?> downloadFile(String content, String name) async {
    try {
      final now = DateTime.now();
      final nowFormatted = DateFormat("yyyy_MM_dd_HH_mm_ss").format(now);
      final appDir = await getApplicationDocumentsDirectory();
      final fileName = "${appDir.path}/${name}_$nowFormatted.txt";
      final file = File(fileName);
      await file.writeAsString(content);
      return fileName;
    } catch (e) {
      return null;
    }
  }
}

FileDownloader getFileDownloader() => NativeFileDownloader();
