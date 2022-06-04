import 'dart:convert' show utf8;
import 'package:collection/collection.dart';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:webapp/base_view.dart';
import 'package:webapp/colors.dart';
import 'package:webapp/components/file_upload.dart';
import 'package:webapp/components/tr_result.dart';
import 'package:webapp/home_model.dart';

Widget wrapScaffold(Widget widget) {
  return SafeArea(child: Scaffold(body: widget));
}

Widget wrapPadding(Widget widget) {
  return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 8),
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

  bool hidePipeline = false;

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
              children: const [
                Text("Failed to retrieve models, please refresh and try again.")
              ],
            ),
          ));
        }
        return SafeArea(
          child: Scaffold(
            appBar: AppBar(
              backgroundColor: uniBlue,
              title: const Text("Spell checking"),
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
            body: Builder(
              builder: (context) {
                Future.delayed(
                  Duration.zero,
                  () {
                    while (model.messageAvailable) {
                      showModalBottomSheetMessage(context, model.popMessage());
                    }
                  },
                );
                return SingleChildScrollView(
                  child: wrapPadding(
                    Column(
                      mainAxisSize: MainAxisSize.min,
                      crossAxisAlignment: CrossAxisAlignment.start,
                      mainAxisAlignment: MainAxisAlignment.start,
                      children: [
                        const Text(
                          "Detect and correct spelling errors in text with or without space errors.",
                          style: TextStyle(fontSize: 22),
                        ),
                        const Divider(height: 32),
                        Row(
                          children: [
                            const Flexible(
                              child: Text(
                                "Select a model pipeline",
                                style: TextStyle(fontSize: 20),
                              ),
                            ),
                            const SizedBox(width: 8),
                            IconButton(
                              onPressed: model.validPipeline
                                  ? () {
                                      setState(() {
                                        hidePipeline = !hidePipeline;
                                      });
                                    }
                                  : null,
                              splashRadius: 16,
                              tooltip: hidePipeline
                                  ? "Show pipeline"
                                  : "Hide pipeline",
                              icon: Icon(hidePipeline
                                  ? Icons.visibility
                                  : Icons.visibility_off),
                            ),
                            const SizedBox(width: 8),
                            IconButton(
                              onPressed: model.validPipeline
                                  ? () {
                                      model.savePipeline();
                                    }
                                  : null,
                              splashRadius: 16,
                              tooltip: "Save pipeline as default",
                              icon: const Icon(Icons.save),
                            )
                          ],
                        ),
                        if (!hidePipeline) const SizedBox(height: 16),
                        if (!hidePipeline)
                          Wrap(
                            runSpacing: 8,
                            children: [
                              Row(
                                children: [
                                  const Icon(Icons.looks_one, color: uniBlue),
                                  Flexible(
                                    child: DropdownButtonFormField<String>(
                                      key: _trModelKey,
                                      value: model.trModel,
                                      decoration: const InputDecoration(
                                          labelText:
                                              "Tokenization repair model"),
                                      icon: const Icon(
                                          Icons.arrow_drop_down_rounded),
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
                                          },
                                        );
                                      },
                                    ),
                                  ),
                                  if (model.trModel != null)
                                    IconButton(
                                      splashRadius: 16,
                                      tooltip:
                                          "Clear tokenization repair model",
                                      color: uniRed,
                                      icon: const Icon(Icons.clear),
                                      onPressed: () {
                                        setState(() {
                                          _trModelKey.currentState!.reset();
                                          model.trModel = null;
                                        });
                                      },
                                    )
                                ],
                              ),
                              Row(
                                children: [
                                  const Icon(Icons.looks_two, color: uniBlue),
                                  Flexible(
                                    child: DropdownButtonFormField<String>(
                                      key: _sedwModelKey,
                                      value: model.sedwModel,
                                      decoration: const InputDecoration(
                                          labelText:
                                              "Word-level spelling error detection model"),
                                      icon: const Icon(
                                          Icons.arrow_drop_down_rounded),
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
                                        });
                                      },
                                    )
                                ],
                              ),
                              Row(
                                children: [
                                  const Icon(Icons.looks_3, color: uniBlue),
                                  Flexible(
                                    child: DropdownButtonFormField<String>(
                                      key: _secModelKey,
                                      value: model.secModel,
                                      decoration: const InputDecoration(
                                          labelText:
                                              "Spelling error correction model"),
                                      icon: const Icon(
                                          Icons.arrow_drop_down_rounded),
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
                                        });
                                      },
                                    )
                                ],
                              ),
                            ],
                          ),
                        const Divider(height: 32),
                        Row(
                          children: [
                            Expanded(
                              child: TextFormField(
                                key: _textKey,
                                maxLines: 10,
                                decoration: InputDecoration(
                                  border: const OutlineInputBorder(),
                                  hintText: "Enter some text to spell check",
                                  suffixIcon: Column(
                                    children: [
                                      IconButton(
                                        icon: const Icon(Icons.clear),
                                        tooltip: "Clear text",
                                        splashRadius: 16,
                                        color: uniRed,
                                        onPressed: model.validPipeline &&
                                                _textKey.currentState?.value !=
                                                    null &&
                                                _textKey.currentState?.value !=
                                                    ""
                                            ? () {
                                                setState(() {
                                                  _textKey.currentState!
                                                      .reset();
                                                });
                                              }
                                            : null,
                                      ),
                                      IconButton(
                                        icon: const Icon(Icons.stream),
                                        color: model.live ? uniBlue : uniGray,
                                        disabledColor: uniGray,
                                        tooltip:
                                            "${!model.live ? "Enable" : "Disable"} live spell checking",
                                        splashRadius: 16,
                                        onPressed: model.validPipeline
                                            ? () {
                                                setState(() {
                                                  model.live = !model.live;
                                                });
                                              }
                                            : null,
                                      )
                                    ],
                                  ),
                                ),
                                enabled: model.validPipeline,
                                onChanged: (String? input) async {
                                  setState(
                                    () {
                                      if (model.live &&
                                          !model.waiting &&
                                          input != null) {
                                        model.inputString = input;
                                        model.runPipeline();
                                      }
                                    },
                                  );
                                },
                              ),
                            ),
                            const SizedBox(
                                width: 32,
                                child: Text(
                                  "or",
                                  textAlign: TextAlign.center,
                                )),
                            Expanded(
                              child: Center(
                                child: FileUpload(
                                  enabled: model.validPipeline && !model.live,
                                  onUpload: (file) {
                                    if (file != null) {
                                      setState(
                                        () {
                                          if (file.extension != "txt") {
                                            model.addMessage(Message(
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
                                          ? () {
                                              setState(
                                                () {
                                                  model.inputString = _textKey
                                                      .currentState?.value;
                                                  model.runPipeline();
                                                },
                                              );
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
                                          ? () {
                                              setState(
                                                () {
                                                  model.inputString =
                                                      model.fileString!;
                                                  model.runPipeline();
                                                },
                                              );
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
                              child: Center(child: CircularProgressIndicator()))
                        ] else ...[
                          if (model.trResults != null)
                            TrResultView(
                                input: model.input,
                                result: model.trResults,
                                onDownload: (fileName) {
                                  setState(() {
                                    if (fileName != null) {
                                      if (!kIsWeb) {
                                        model.addMessage(
                                          Message(
                                              "Downloaded tokenization repair outputs to $fileName",
                                              Status.info),
                                        );
                                      }
                                    } else {
                                      model.addMessage(Message(
                                          "Unexpected error downloading tokenization repair outputs",
                                          Status.error));
                                    }
                                  });
                                }),
                          if (model.sedwResults != null)
                            Text(
                                model.sedwResults["output"]["text"].join("\n")),
                          if (model.secResults != null)
                            Text(model.secResults["output"]["text"].join("\n"))
                        ],
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
        );
      },
    );
  }

  void showModalBottomSheetMessage(BuildContext context, Message message) {
    bool popped = false;
    Scaffold.of(context).showBottomSheet(
      (context) {
        Future.delayed(const Duration(seconds: 3), () {
          if (!popped) {
            Navigator.of(context).pop(true);
          }
        });
        return Container(
          margin: const EdgeInsets.all(10),
          decoration: BoxDecoration(
            borderRadius: const BorderRadius.all(Radius.circular(5)),
            color: message.status == Status.info
                ? uniGreen
                : message.status == Status.warn
                    ? uniOrange
                    : uniRed,
          ),
          child: GestureDetector(
            behavior: HitTestBehavior.opaque,
            onTap: () {
              popped = true;
              Navigator.of(context).pop();
            },
            child: Padding(
              padding: const EdgeInsets.all(15),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Text(
                    message.message,
                    style: const TextStyle(color: Colors.white),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}
