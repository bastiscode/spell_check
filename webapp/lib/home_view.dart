import 'dart:convert' show utf8;
import 'package:collection/collection.dart';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:webapp/base_view.dart';
import 'package:webapp/colors.dart';
import 'package:webapp/components/file_upload.dart';
import 'package:webapp/components/message.dart';
import 'package:webapp/components/result.dart';
import 'package:webapp/home_model.dart';
import 'package:webapp/utils.dart';

Widget wrapScaffold(Widget widget) {
  return SafeArea(child: Scaffold(body: widget));
}

Widget wrapPadding(Widget widget) {
  return Padding(
      padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 16),
      child: widget);
}

class HomeView extends StatefulWidget {
  const HomeView({super.key});

  @override
  State<HomeView> createState() => _HomeViewState();
}

class _HomeViewState extends State<HomeView> {
  final GlobalKey<FormFieldState> _trModelKey = GlobalKey();
  final GlobalKey<FormFieldState> _sedwModelKey = GlobalKey();
  final GlobalKey<FormFieldState> _secModelKey = GlobalKey();
  final GlobalKey<FormFieldState> _textKey = GlobalKey();
  final TextEditingController _filterController = TextEditingController();

  @override
  Widget build(BuildContext homeContext) {
    return BaseView<HomeModel>(
      onModelReady: (model) async {
        await model.init();
      },
      builder: (context, model, child) {
        if (!model.ready) {
          return wrapScaffold(
              const Center(child: CircularProgressIndicator(color: uniBlue)));
        } else if (model.ready && !model.available) {
          return wrapScaffold(Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                const Text(
                    "Failed to retrieve models, please try again and reload."),
                const SizedBox(
                  height: 8,
                ),
                ElevatedButton.icon(
                    onPressed: () async {
                      await model.init();
                    },
                    icon: const Icon(Icons.refresh),
                    label: const Text("Reload"))
              ],
            ),
          ));
        }
        return SafeArea(
          child: Scaffold(
            appBar: AppBar(
              backgroundColor: uniBlue,
              title: const Text("Spell checking"),
              centerTitle: true,
              actions: [
                IconButton(
                  icon: const Icon(Icons.info),
                  tooltip: "Show backend information",
                  onPressed: () {
                    showDialog(
                      context: context,
                      builder: (infoContext) {
                        List<dynamic> gpus = model.info["gpu"];
                        return SimpleDialog(
                          title: const Text("Info"),
                          children: [
                            if (model.info == null) ...[
                              const SimpleDialogOption(
                                  child: Text("Info not available"))
                            ] else ...[
                              SimpleDialogOption(
                                  child: Text(
                                      "Library: nsc (version ${model.info["version"]})")),
                              SimpleDialogOption(
                                  child: Text(
                                      "Timeout: ${model.info["timeout"]} seconds")),
                              SimpleDialogOption(
                                  child: Text(
                                      "Precision: ${model.info["precision"]}")),
                              SimpleDialogOption(
                                  child: Text("CPU: ${model.info["cpu"]}")),
                              SimpleDialogOption(
                                child: Row(
                                  children: [
                                    const Text("GPUs: "),
                                    if (gpus.isEmpty)
                                      const Text("-")
                                    else
                                      Flexible(
                                        child: Wrap(
                                          children: gpus
                                              .mapIndexed(
                                                (idx, gpu) => Card(
                                                  child: Padding(
                                                    padding:
                                                        const EdgeInsets.all(8),
                                                    child:
                                                        Text("GPU $idx: $gpu"),
                                                  ),
                                                ),
                                              )
                                              .toList(),
                                        ),
                                      )
                                  ],
                                ),
                              )
                            ],
                          ],
                        );
                      },
                    );
                  },
                )
              ],
            ),
            body: SingleChildScrollView(
              child: wrapPadding(
                Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    const Text(
                      "Detect and correct spelling errors in text with or without space errors.",
                      style: TextStyle(fontSize: 22),
                    ),
                    const SizedBox(height: 16),
                    ExpansionTile(
                      tilePadding: EdgeInsets.zero,
                      initiallyExpanded: !model.validPipeline,
                      controlAffinity: ListTileControlAffinity.leading,
                      title: const Text(
                        "Select a model pipeline",
                        style: TextStyle(fontSize: 20),
                      ),
                      subtitle: const Text(
                          "At least one model must be selected for a valid pipeline"),
                      childrenPadding: EdgeInsets.zero,
                      children: [
                        const SizedBox(height: 16),
                        Row(
                          children: [
                            const Icon(Icons.looks_one, color: uniBlue),
                            const SizedBox(width: 8),
                            Flexible(
                              child: DropdownButtonFormField<String>(
                                key: _trModelKey,
                                value: model.trModel,
                                decoration: const InputDecoration(
                                    labelText: "Tokenization repair model"),
                                icon: const Icon(Icons.arrow_drop_down_rounded),
                                items: model
                                    .getModels("tokenization repair")
                                    .map<DropdownMenuItem<String>>(
                                  (modelInfo) {
                                    return DropdownMenuItem(
                                      value: modelInfo["name"],
                                      child: Text(modelInfo["name"]),
                                    );
                                  },
                                ).toList(),
                                onChanged: (String? modelName) {
                                  setState(
                                    () {
                                      model.trModel = modelName!;
                                      model.lastInputString = "";
                                    },
                                  );
                                },
                              ),
                            ),
                            if (model.trModel != null)
                              IconButton(
                                splashRadius: 16,
                                tooltip: "Clear tokenization repair model",
                                color: uniRed,
                                icon: const Icon(Icons.clear),
                                onPressed: () {
                                  setState(() {
                                    _trModelKey.currentState!.reset();
                                    model.trModel = null;
                                    model.lastInputString = "";
                                  });
                                },
                              )
                          ],
                        ),
                        const SizedBox(height: 8),
                        Row(
                          children: [
                            const Icon(Icons.looks_two, color: uniBlue),
                            const SizedBox(width: 8),
                            Flexible(
                              child: DropdownButtonFormField<String>(
                                key: _sedwModelKey,
                                value: model.sedwModel,
                                decoration: const InputDecoration(
                                    labelText:
                                        "Word-level spelling error detection model"),
                                icon: const Icon(Icons.arrow_drop_down_rounded),
                                items: model
                                    .getModels("sed words")
                                    .map<DropdownMenuItem<String>>(
                                  (modelInfo) {
                                    return DropdownMenuItem(
                                      value: modelInfo["name"],
                                      child: Text(modelInfo["name"]),
                                    );
                                  },
                                ).toList(),
                                onChanged: (String? modelName) {
                                  setState(
                                    () {
                                      model.sedwModel = modelName!;
                                      model.lastInputString = "";
                                    },
                                  );
                                },
                              ),
                            ),
                            if (model.sedwModel != null)
                              IconButton(
                                splashRadius: 16,
                                tooltip:
                                    "Clear word-level spelling error detection model",
                                color: uniRed,
                                icon: const Icon(Icons.clear),
                                onPressed: () {
                                  setState(() {
                                    _sedwModelKey.currentState!.reset();
                                    model.sedwModel = null;
                                    model.lastInputString = "";
                                  });
                                },
                              )
                          ],
                        ),
                        const SizedBox(height: 8),
                        Row(
                          children: [
                            const Icon(Icons.looks_3, color: uniBlue),
                            const SizedBox(width: 8),
                            Flexible(
                              child: DropdownButtonFormField<String>(
                                key: _secModelKey,
                                value: model.secModel,
                                decoration: const InputDecoration(
                                    labelText:
                                        "Spelling error correction model"),
                                icon: const Icon(Icons.arrow_drop_down_rounded),
                                items: model
                                    .getModels("sec")
                                    .map<DropdownMenuItem<String>>(
                                  (modelInfo) {
                                    return DropdownMenuItem(
                                      value: modelInfo["name"],
                                      child: Text(modelInfo["name"]),
                                    );
                                  },
                                ).toList(),
                                onChanged: (String? modelName) {
                                  setState(() {
                                    model.secModel = modelName!;
                                    model.lastInputString = "";
                                  });
                                },
                              ),
                            ),
                            if (model.secModel != null)
                              IconButton(
                                splashRadius: 16,
                                tooltip:
                                    "Clear spelling error correction model",
                                color: uniRed,
                                icon: const Icon(Icons.clear),
                                onPressed: () {
                                  setState(() {
                                    _secModelKey.currentState!.reset();
                                    model.secModel = null;
                                    model.lastInputString = "";
                                  });
                                },
                              )
                          ],
                        ),
                        const SizedBox(height: 8),
                        Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              ElevatedButton.icon(
                                onPressed: model.validPipeline
                                    ? () async {
                                        await model.savePipeline();
                                        setState(() {
                                          showMessage(
                                              context,
                                              Message("Saved pipeline settings",
                                                  Status.info));
                                        });
                                      }
                                    : null,
                                icon: const Icon(Icons.save),
                                label: const Text("Save"),
                              ),
                            ]),
                        const SizedBox(height: 16),
                      ],
                    ),
                    const SizedBox(height: 16),
                    Row(
                      children: [
                        Expanded(
                          child: TextFormField(
                            key: _textKey,
                            maxLines: model.live ? 1 : 5,
                            decoration: InputDecoration(
                              border: const OutlineInputBorder(),
                              hintText: model.live
                                  ? "Enter some text to spell check while typing..."
                                  : "Enter some text to spell check...",
                              suffixIcon: Column(
                                children: [
                                  IconButton(
                                    icon: const Icon(Icons.clear),
                                    tooltip: "Clear text",
                                    splashRadius: 16,
                                    color: uniRed,
                                    onPressed: model.validPipeline &&
                                            !model.waiting &&
                                            (_textKey.currentState?.value ??
                                                    "") !=
                                                ""
                                        ? () {
                                            setState(() {
                                              _textKey.currentState!
                                                  .didChange("");
                                              model.inputString = "";
                                            });
                                          }
                                        : null,
                                  ),
                                  if (!model.live)
                                    IconButton(
                                      icon: const Icon(Icons.paste),
                                      tooltip: "Paste text from clipboard",
                                      splashRadius: 16,
                                      onPressed: model.validPipeline &&
                                              !model.waiting
                                          ? () async {
                                              final data =
                                                  await Clipboard.getData(
                                                      "text/plain");
                                              setState(() {
                                                if (data != null) {
                                                  _textKey.currentState!
                                                      .didChange(data.text!);
                                                  model.inputString =
                                                      data.text!;
                                                } else {
                                                  showMessage(
                                                      context,
                                                      Message(
                                                          "could not paste text from clipboard",
                                                          Status.error));
                                                }
                                              });
                                            }
                                          : null,
                                    ),
                                  IconButton(
                                    icon: const Icon(Icons.stream),
                                    color: model.live ? uniBlue : uniDarkGray,
                                    tooltip:
                                        "${!model.live ? "Enable" : "Disable"} live spell checking",
                                    splashRadius: 16,
                                    onPressed: model.validPipeline &&
                                            !model.waiting
                                        ? () async {
                                            setState(
                                              () {
                                                model.live = !model.live;
                                              },
                                            );
                                            if (model.live) {
                                              String currentText = _textKey
                                                      .currentState!.value ??
                                                  "";
                                              currentText =
                                                  currentText.split("\n")[0];
                                              _textKey.currentState!
                                                  .didChange(currentText);
                                              model.inputString = currentText;
                                              await model.runPipelineLive();
                                            }
                                          }
                                        : null,
                                  ),
                                ],
                              ),
                            ),
                            enabled: model.validPipeline,
                            onChanged: (String? input) async {
                              if (model.live) {
                                model.inputString = input ?? "";
                              }
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
                              enabled: model.validPipeline && !model.live,
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
                                        String fileString =
                                            utf8.decode(file.bytes!);
                                        model.fileString = fileString;
                                      }
                                    },
                                  );
                                }
                              },
                              onError: (message) {
                                showMessage(
                                    context, Message(message, Status.error));
                              },
                              onDelete: () {
                                setState(() {
                                  model.fileString = null;
                                });
                              },
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(
                      height: 8,
                    ),
                    Row(
                      children: [
                        Expanded(
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Flexible(
                                child: ElevatedButton.icon(
                                  onPressed: model.validPipeline &&
                                          !model.waiting &&
                                          !model.live
                                      ? () async {
                                          model.inputString =
                                              _textKey.currentState?.value;
                                          final error =
                                              await model.runPipeline();
                                          if (error != null && mounted) {
                                            showMessage(context, error);
                                          }
                                        }
                                      : null,
                                  icon: const Icon(Icons.edit_note),
                                  label: const Text("Run pipeline on text"),
                                ),
                              )
                            ],
                          ),
                        ),
                        const SizedBox(
                          width: 32,
                        ),
                        Expanded(
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Flexible(
                                child: ElevatedButton.icon(
                                  onPressed: model.validPipeline &&
                                          !model.waiting &&
                                          !model.live &&
                                          model.fileString != null
                                      ? () async {
                                          model.inputString = model.fileString!;
                                          final error =
                                              await model.runPipeline();
                                          if (error != null && mounted) {
                                            showMessage(context, error);
                                          }
                                        }
                                      : null,
                                  icon: const Icon(Icons.edit_note),
                                  label: const Text("Run pipeline on file"),
                                ),
                              )
                            ],
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),
                    if (model.waiting && !model.live) ...[
                      const Flexible(
                        fit: FlexFit.loose,
                        child: Center(
                          child: CircularProgressIndicator(),
                        ),
                      )
                    ] else if (model.hasResults) ...[
                      const SizedBox(height: 8),
                      ListTile(
                        contentPadding: EdgeInsets.zero,
                        title: const Text(
                          "Results",
                          style: TextStyle(fontSize: 20),
                        ),
                        subtitle: Text(
                            "${formatS(model.runtimes["total"]["s"])}, ${formatB(model.runtimes["total"]["bps"])}/s"),
                      ),
                      ResultView(
                        input: model.input,
                        output: model.outputs,
                        runtimes: model.runtimes,
                        onClipboard: (name) {
                          showMessage(
                              context,
                              Message("Copied $name outputs to clipboard",
                                  Status.info));
                        },
                        onDownload: (fileName, name) {
                          if (fileName != null) {
                            showMessage(
                              context,
                              Message("Downloaded $name outputs to $fileName",
                                  Status.info),
                            );
                          } else {
                            showMessage(
                                context,
                                Message(
                                    "Unexpected error downloading $name outputs",
                                    Status.error));
                          }
                        },
                      ),
                      ExpansionTile(
                        tilePadding: EdgeInsets.zero,
                          maintainState: true,
                          childrenPadding: EdgeInsets.zero,
                          controlAffinity: ListTileControlAffinity.leading,
                          title: const Text(
                            "Evaluation",
                            style: TextStyle(fontSize: 20),
                          ),
                          subtitle: Text(
                              "Evaluate the results from above against their groundtruths"),
                          children: [])
                    ],
                  ],
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}
