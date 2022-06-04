import "package:webapp/platforms/file_download_stub.dart"
    if (dart.library.io) "package:webapp/platforms/file_download_native.dart"
    if (dart.library.html) "package:webapp/platforms/file_download_web.dart";

abstract class FileDownloader {
  Future<String?> downloadFile(String content, String name) async {
    throw UnimplementedError();
  }

  factory FileDownloader() => getFileDownloader();
}
