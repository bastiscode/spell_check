import 'dart:convert' show utf8;

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:webapp/colors.dart';

typedef OnUploadCallback = Function(PlatformFile? file);
typedef OnDeleteCallback = Function();

class FileUpload extends StatefulWidget {
  final bool enabled;
  final OnUploadCallback onUpload;
  final OnDeleteCallback onDelete;

  const FileUpload(
      {required this.enabled,
      required this.onUpload,
      required this.onDelete,
      super.key});

  @override
  State<FileUpload> createState() => _FileUploadState();
}

class _FileUploadState extends State<FileUpload> {
  PlatformFile? _file;

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      child: _file == null
          ? ElevatedButton.icon(
              onPressed: widget.enabled
                  ? () async {
                      final files = await FilePicker.platform.pickFiles(
                          dialogTitle: "Pick a text file",
                          type: FileType.custom,
                          allowedExtensions: ["txt"]);
                      if (files != null) {
                        _file = files.files.single;
                        widget.onUpload(_file);
                      } else {
                        widget.onUpload(null);
                      }
                    }
                  : null,
              icon: const Icon(Icons.upload_file),
              label: const Text("Upload a text file"),
            )
          : Card(
              child: ListTile(
                title: const Text("Upload successful"),
                subtitle: Text(
                    "File ${_file!.name} contains ${_file!.bytes!.length / 1000}kB of text in ${utf8.decode(_file!.bytes!).split("\n").length} lines."),
                trailing: IconButton(
                  tooltip: "Delete uploaded file",
                  onPressed: () {
                    setState(
                      () {
                        _file = null;
                        widget.onDelete();
                      },
                    );
                  },
                  splashRadius: 16,
                  icon: const Icon(
                    Icons.delete,
                    color: uniRed,
                  ),
                ),
              ),
            ),
    );
  }
}
