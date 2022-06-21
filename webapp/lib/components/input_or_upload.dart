import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:webapp/colors.dart';
import 'package:webapp/components/file_upload.dart';
import 'package:webapp/components/message.dart';

typedef OnSubmitCallback = Function(String input);

class InputOrUpload extends StatefulWidget {
  final String title;
  final OnSubmitCallback onSubmit;

  const InputOrUpload({required this.title, required this.onSubmit, super.key});

  @override
  State<StatefulWidget> createState() => _InputOrUploadState();
}

class _InputOrUploadState extends State<InputOrUpload> {
  final GlobalKey<FormFieldState> _textKey = GlobalKey();
  String? inputString;
  String? inputFile;

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      actionsAlignment: MainAxisAlignment.center,
      title: Text(
        widget.title,
        textAlign: TextAlign.center,
      ),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Row(
            mainAxisSize: MainAxisSize.max,
            children: [
              Expanded(
                child: TextFormField(
                    key: _textKey,
                    maxLines: 10,
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
                            onPressed: (_textKey.currentState?.value ?? "") !=
                                    ""
                                ? () {
                                    setState(() {
                                      _textKey.currentState!.didChange(null);
                                      inputString = null;
                                    });
                                  }
                                : null,
                          ),
                          IconButton(
                            icon: const Icon(Icons.paste),
                            tooltip: "Paste text from clipboard",
                            splashRadius: 16,
                            onPressed: () async {
                              final data =
                                  await Clipboard.getData("text/plain");
                              if (data != null) {
                                setState(() {
                                  inputString = data.text!;
                                  _textKey.currentState!.didChange(inputString);
                                });
                              }
                            },
                          )
                        ],
                      ),
                    )),
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
                              inputFile = utf8.decode(file.bytes!);
                            }
                          },
                        );
                      }
                    },
                    onError: (message) {
                      showMessage(context, Message(message, Status.error));
                    },
                    onDelete: () {
                      setState(() {
                        inputFile = null;
                      });
                    },
                  ),
                ),
              ),
            ],
          )
        ],
      ),
      actions: [
        ElevatedButton.icon(
            style: ElevatedButton.styleFrom(primary: uniDarkGray),
            onPressed: () {
              Navigator.of(context).pop();
            },
            icon: const Icon(Icons.cancel),
            label: const Text("Cancel")),
        ElevatedButton.icon(
            onPressed: inputString != null || inputFile != null
                ? () {
                    if (inputString != null && inputFile != null) {
                      showMessage(
                          context,
                          Message(
                              "either enter some text or upload a file, but not both at the same time",
                              Status.warn));
                    } else if (inputString != null) {
                      widget.onSubmit(inputString!);
                      Navigator.of(context).pop();
                    } else if (inputFile != null) {
                      widget.onSubmit(inputFile!);
                      Navigator.of(context).pop();
                    }
                  }
                : null,
            icon: const Icon(Icons.done),
            label: const Text("Apply"))
      ],
    );
  }
}

Future<String?> showInputOrUploadDialog(
    BuildContext context, String title) async {
  String? text;
  await showDialog(
    context: context,
    builder: (dialogContext) {
      return ScaffoldMessenger(
        child: Builder(
          builder: (context) => Scaffold(
            backgroundColor: Colors.transparent,
            body: InputOrUpload(
                title: title,
                onSubmit: (submitted) {
                  text = submitted;
                }),
          ),
        ),
      );
    },
  );
  return text;
}
