import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:webapp/colors.dart';
import 'package:webapp/components/file_upload.dart';
import 'package:webapp/components/message.dart';

typedef OnTextCallback = Function(String?);

class InputOrUpload extends StatefulWidget {
  final OnTextCallback onInputChange;
  final OnTextCallback onUploadChange;

  const InputOrUpload(
      {required this.onInputChange, required this.onUploadChange, super.key});

  @override
  State<StatefulWidget> createState() => _InputOrUploadState();
}

class _InputOrUploadState extends State<InputOrUpload> {
  final GlobalKey<FormFieldState> _textKey = GlobalKey();

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        Row(
          mainAxisSize: MainAxisSize.max,
          children: [
            Expanded(
              child: TextFormField(
                key: _textKey,
                maxLines: 5,
                decoration: InputDecoration(
                  border: const OutlineInputBorder(),
                  hintText: "Enter text here...",
                  suffixIcon: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      IconButton(
                        icon: const Icon(Icons.clear),
                        tooltip: "Clear text",
                        splashRadius: 16,
                        color: uniRed,
                        onPressed: (_textKey.currentState?.value ?? "") != ""
                            ? () {
                                setState(() {
                                  _textKey.currentState!.didChange(null);
                                  widget.onInputChange(null);
                                });
                              }
                            : null,
                      ),
                      IconButton(
                        icon: const Icon(Icons.paste),
                        tooltip: "Paste text from clipboard",
                        splashRadius: 16,
                        onPressed: () async {
                          final data = await Clipboard.getData("text/plain");
                          if (data != null) {
                            setState(() {
                              _textKey.currentState!.didChange(data.text!);
                              widget.onInputChange(data.text!);
                            });
                          }
                        },
                      )
                    ],
                  ),
                ),
                onChanged: (value) {
                  widget.onInputChange(value);
                },
              ),
            ),
            const SizedBox(
              width: 32,
              child: Text(
                "or",
                textAlign: TextAlign.center,
              ),
            ),
            Expanded(
              child: Center(
                child: FileUpload(
                  enabled: true,
                  onUpload: (file) {
                    if (file != null) {
                      setState(
                        () {
                          if (file.extension != "txt") {
                            showMessage(
                                context,
                                Message(
                                    "the uploaded file must be a utf-8 encoded text file (.txt), but got extension '${file.extension}'",
                                    Status.warn));
                          } else {
                            widget.onUploadChange(utf8.decode(file.bytes!));
                          }
                        },
                      );
                    }
                  },
                  onError: (message) {
                    showMessage(context, Message(message, Status.error));
                  },
                  onDelete: () {
                    widget.onUploadChange(null);
                  },
                ),
              ),
            ),
          ],
        ),
      ],
    );
  }
}

Future<String?> showInputOrUploadDialog(
    BuildContext context, String title) async {
  String? textInput;
  String? fileInput;
  String? output;
  await showDialog(
    context: context,
    builder: (dialogContext) {
      return ScaffoldMessenger(
        child: StatefulBuilder(
          builder: (context, setState) => Scaffold(
            backgroundColor: Colors.transparent,
            body: AlertDialog(
              title: Text(title),
              content: InputOrUpload(
                onInputChange: (text) {
                  setState(() {
                    textInput = text;
                  });
                },
                onUploadChange: (text) {
                  setState(() {
                    fileInput = text;
                  });
                },
              ),
              actions: [
                ElevatedButton.icon(
                    onPressed: () {
                      output = null;
                      Navigator.of(context).pop();
                    },
                    style: ElevatedButton.styleFrom(primary: uniDarkGray),
                    icon: const Icon(Icons.cancel),
                    label: const Text("Close")),
                ElevatedButton.icon(
                  onPressed: textInput != null || fileInput != null
                      ? () {
                          if (textInput != null && fileInput != null) {
                            showMessage(
                              context,
                              Message(
                                  "either enter some text or upload a file, but not both at the same time",
                                  Status.warn),
                            );
                          } else if (textInput != null) {
                            output = textInput;
                            Navigator.of(context).pop();
                          } else if (fileInput != null) {
                            output = fileInput;
                            Navigator.of(context).pop();
                          }
                        }
                      : null,
                  icon: const Icon(Icons.done),
                  label: const Text("Apply"),
                )
              ],
            ),
          ),
        ),
      );
    },
  );
  return output;
}
