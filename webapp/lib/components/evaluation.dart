import 'dart:convert';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:webapp/colors.dart';
import 'package:webapp/components/message.dart';
import 'package:webapp/utils.dart';

typedef EvaluationCallback = Future<Widget?> Function(String);

class EvaluationView extends StatefulWidget {
  final int inputLength;
  final EvaluationCallback onEvaluation;
  final String taskName;

  const EvaluationView(
      {required this.inputLength,
      required this.onEvaluation,
      required this.taskName,
      super.key});

  @override
  State<StatefulWidget> createState() => _EvaluationViewState();
}

class _EvaluationViewState extends State<EvaluationView> {
  final inputController = TextEditingController();
  bool evaluating = false;
  Widget? evaluation;

  @override
  void initState() {
    super.initState();

    inputController.addListener(() {
      setState(() {});
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        ListTile(
          contentPadding: EdgeInsets.zero,
          visualDensity: VisualDensity.compact,
          title: Text("Evaluate ${widget.taskName}"),
          subtitle: Text(
              "Specify a well-formed ${widget.taskName} groundtruth with the same number of lines as the output above"),
        ),
        TextField(
          controller: inputController,
          maxLines: 10,
          readOnly: evaluating,
          decoration: InputDecoration(
            border: const OutlineInputBorder(),
            hintText: "Enter ${widget.taskName} groundtruth...",
            helperText: helperTextFromTextController(inputController),
            suffixIcon: Column(
              children: [
                IconButton(
                  icon: const Icon(Icons.clear),
                  tooltip: "Clear groundtruth",
                  splashRadius: 16,
                  color: uniRed,
                  onPressed: inputController.text.isNotEmpty && !evaluating
                      ? () {
                          setState(
                            () {
                              inputController.value = const TextEditingValue(
                                text: "",
                                selection: TextSelection.collapsed(offset: 0),
                              );
                            },
                          );
                        }
                      : null,
                ),
                IconButton(
                  icon: const Icon(Icons.paste),
                  tooltip: "Paste groundtruth from clipboard",
                  splashRadius: 16,
                  onPressed: !evaluating
                      ? () async {
                          final data = await Clipboard.getData("text/plain");
                          setState(
                            () {
                              if (data != null) {
                                inputController.value = TextEditingValue(
                                  text: data.text!,
                                  selection: TextSelection.collapsed(
                                      offset: data.text!.length),
                                );
                              } else {
                                showMessage(
                                  context,
                                  Message(
                                      "could not paste groundtruth from clipboard",
                                      Status.error),
                                );
                              }
                            },
                          );
                        }
                      : null,
                ),
                IconButton(
                  onPressed: !evaluating
                      ? () async {
                          try {
                            final files = await FilePicker.platform.pickFiles(
                                dialogTitle: "Pick a text file",
                                type: FileType.custom,
                                allowedExtensions: ["txt"]);
                            if (files != null) {
                              final file = files.files.single;
                              final content = utf8.decode(file.bytes!);
                              setState(
                                () {
                                  inputController.value = TextEditingValue(
                                    text: content,
                                    selection: TextSelection.collapsed(
                                        offset: content.length),
                                  );
                                },
                              );
                            }
                          } on FormatException catch (_) {
                            showMessage(
                              context,
                              Message(
                                  "error decoding file, make sure that it contains valid utf8 bytes",
                                  Status.error),
                            );
                          } on PlatformException catch (e) {
                            showMessage(
                              context,
                              Message("error uploading file: ${e.message}",
                                  Status.error),
                            );
                          }
                        }
                      : null,
                  icon: const Icon(Icons.upload_file),
                  tooltip: "Upload the groundtruth file",
                  splashRadius: 16,
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 8),
        evaluating
            ? const SizedBox(
                height: 16,
                width: 16,
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                ),
              )
            : ElevatedButton.icon(
                onPressed: inputController.text.split("\n").length ==
                        widget.inputLength
                    ? () async {
                        setState(
                          () {
                            evaluation = null;
                            evaluating = true;
                          },
                        );
                        evaluation =
                            await widget.onEvaluation(inputController.text);
                        setState(() {
                          evaluating = false;
                        });
                      }
                    : null,
                icon: const Icon(Icons.analytics),
                label: const Text("Evaluate"),
              ),
        const SizedBox(height: 8),
        if (evaluation != null) evaluation!
      ],
    );
  }
}
