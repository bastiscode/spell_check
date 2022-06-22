import 'dart:convert' show utf8;

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:webapp/colors.dart';
import 'package:webapp/utils.dart';

typedef OnUploadCallback = Function(PlatformFile?);
typedef OnErrorCallback = Function(String);

class FileUpload extends StatefulWidget {
  final bool enabled;
  final OnUploadCallback onUpload;
  final OnErrorCallback onError;
  final VoidCallback onDelete;

  const FileUpload(
      {required this.enabled,
      required this.onUpload,
      required this.onError,
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
                      try {
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
                      } on PlatformException catch (e) {
                        widget.onError("error uploading file: ${e.message}");
                      }
                    }
                  : null,
              icon: const Icon(Icons.upload_file),
              label: const Text("Upload a text file"),
            )
          : Uploaded(
              title: "Upload successful",
              name: "File ${_file!.name}",
              bytes: _file!.bytes!.length,
              lines: utf8.decode(_file!.bytes!).split("\n").length,
              onDelete: () {
                _file = null;
                widget.onDelete();
              },
            ),
    );
  }
}

class Uploaded extends StatefulWidget {
  final String title;
  final String name;
  final int bytes;
  final int lines;
  final VoidCallback onDelete;

  const Uploaded(
      {required this.title,
      required this.name,
      required this.bytes,
      required this.lines,
      required this.onDelete,
      super.key});

  @override
  State createState() => _UploadedState();
}

class _UploadedState extends State<Uploaded> {
  @override
  Widget build(BuildContext context) {
    return Card(
      child: ListTile(
        visualDensity: VisualDensity.compact,
        title: Text(widget.title),
        subtitle: Text(
            "${widget.name} contains ${formatB(widget.bytes.toDouble())} of text in ${widget.lines} lines."),
        trailing: IconButton(
          tooltip: "Delete uploaded file",
          onPressed: widget.onDelete,
          splashRadius: 16,
          icon: const Icon(
            Icons.delete,
            color: uniRed,
          ),
        ),
      ),
    );
  }
}
