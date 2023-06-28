import 'dart:convert' show utf8;
import 'package:collection/collection.dart';
import 'package:file_picker/file_picker.dart';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:webapp/api.dart';
import 'package:webapp/base_view.dart';
import 'package:webapp/colors.dart';
import 'package:webapp/components/evaluation.dart';
import 'package:webapp/components/message.dart';
import 'package:webapp/components/presets.dart';
import 'package:webapp/components/result.dart';
import 'package:webapp/home_model.dart';
import 'package:webapp/platforms/file_download_interface.dart';
import 'package:webapp/utils.dart';

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
  final TextEditingController inputController = TextEditingController();
  final TextEditingController outputController = TextEditingController();

  bool showPipelineInfo = false;

  @override
  void initState() {
    super.initState();

    inputController.addListener(() {
      setState(() {});
    });
  }

  @override
  Widget build(BuildContext homeContext) {
    return BaseView<HomeModel>(
      onModelReady: (model) async {
        await model.init(inputController, outputController);
      },
      builder: (context, model, child) {
        if (!model.ready) {
          return wrapScaffold(
            const Center(
              child: CircularProgressIndicator(
                strokeWidth: 2,
              ),
            ),
          );
        } else if (model.ready && !model.available) {
          return wrapScaffold(
            Center(
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
                        await model.init(inputController, outputController);
                      },
                      icon: const Icon(Icons.refresh),
                      label: const Text("Reload"))
                ],
              ),
            ),
          );
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
                    showInfoDialog(model.info);
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
                    Card(
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(8)),
                      elevation: 2,
                      child: Padding(
                        padding: const EdgeInsets.symmetric(
                            vertical: 8, horizontal: 16),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          mainAxisSize: MainAxisSize.min,
                          children: const [
                            ListTile(
                              visualDensity: VisualDensity.compact,
                              contentPadding: EdgeInsets.zero,
                              title: Text(
                                "Detect and correct spelling errors in text with or without space errors.",
                                style: TextStyle(fontSize: 22),
                              ),
                              // subtitle: Text("Last updated: Jun 22, 2022"),
                            )
                          ],
                        ),
                      ),
                    ),
                    buildInputOutput(model),
                    buildPipeline(model),
                    if (model.hasResults) ...[
                      Card(
                        shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(8)),
                        elevation: 2,
                        clipBehavior: Clip.antiAlias,
                        child: ExpansionTile(
                          maintainState: true,
                          controlAffinity: ListTileControlAffinity.leading,
                          title: const Text(
                            "Detailed outputs",
                            style: TextStyle(fontSize: 20),
                          ),
                          subtitle: const Text(
                              "Inspect the ouputs of all pipeline steps"),
                          children: [
                            ResultView(
                                input: model.input,
                                output: model.outputs,
                                runtimes: model.runtimes),
                          ],
                        ),
                      ),
                      buildEvaluation(model)
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

  Widget buildPipeline(HomeModel model) {
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      elevation: 2,
      clipBehavior: Clip.antiAlias,
      child: ExpansionTile(
        maintainState: true,
        initiallyExpanded: !model.validPipeline,
        controlAffinity: ListTileControlAffinity.leading,
        title: const Text(
          "Select a model pipeline",
          style: TextStyle(fontSize: 20),
        ),
        subtitle: const Text(
            "The pipeline determines how the input text will be spell checked"),
        childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 8),
        expandedCrossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const SizedBox(height: 8),
          const Text(
              "Select at least one model for any of the three tasks below to form a valid pipeline. Once you selected a valid pipeline you can run it on "
              "some text. Alternatively choose one of the presets below depending on your use case. "
              "You can also save the current pipeline settings to avoid specifying them each time you visit this site."),
          const SizedBox(height: 16),
          Presets(
            presets: const [
              Preset("Best full pipeline",
                  trModel: "eo large arxiv with errors",
                  sedwModel: "gnn+",
                  secModel: "transformer nmt"),
              Preset("Fast full pipeline",
                  trModel: "eo large arxiv with errors",
                  sedwModel: "transformer+",
                  secModel: "transformer words nmt"),
              Preset("Best all in one", secModel: "tokenization repair++"),
              Preset("Best for highly corrupted text",
                  secModel: "transformer with tokenization repair nmt"),
              Preset("Whitespace correction only",
                  trModel: "eo large arxiv with errors"),
              Preset("Error detection only", sedwModel: "gnn+"),
              Preset("Error correction only", secModel: "transformer words nmt")
            ],
            models: model.models,
            trModel: model.trModel,
            sedwModel: model.sedwModel,
            secModel: model.secModel,
            onSelected: (preset) {
              setState(
                () {
                  model.lastInputString = null;
                  if (preset == null) {
                    model.trModel = null;
                    model.sedwModel = null;
                    model.secModel = null;
                  } else {
                    model.trModel = preset.trModel;
                    model.sedwModel = preset.sedwModel;
                    model.secModel = preset.secModel;
                  }
                },
              );
            },
          ),
          const SizedBox(height: 16),
          DropdownButtonFormField<String>(
            value: model.trModel,
            isExpanded: true,
            decoration: InputDecoration(
                prefixIcon: const Icon(Icons.looks_one, color: uniBlue),
                suffixIcon: IconButton(
                  splashRadius: 16,
                  tooltip: "Clear tokenization repair model",
                  color: uniRed,
                  icon: const Icon(Icons.clear),
                  onPressed: model.trModel != null
                      ? () {
                          setState(() {
                            model.trModel = null;
                            model.lastInputString = null;
                          });
                        }
                      : null,
                ),
                labelText: "Tokenization repair model",
                helperMaxLines: 10,
                helperText: model.trModel != null
                    ? model.getModel(
                        "tokenization repair", model.trModel!)["description"]
                    : null),
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
                  model.lastInputString = null;
                },
              );
            },
          ),
          const SizedBox(height: 16),
          DropdownButtonFormField<String>(
            value: model.sedwModel,
            isExpanded: true,
            decoration: InputDecoration(
                prefixIcon: const Icon(Icons.looks_two, color: uniBlue),
                suffixIcon: IconButton(
                  splashRadius: 16,
                  tooltip: "Clear spelling error detection model",
                  color: uniRed,
                  icon: const Icon(Icons.clear),
                  onPressed: model.sedwModel != null
                      ? () {
                          setState(() {
                            model.sedwModel = null;
                            model.lastInputString = null;
                          });
                        }
                      : null,
                ),
                labelText: "Word-level spelling error detection model",
                helperMaxLines: 10,
                helperText: model.sedwModel != null
                    ? model.getModel(
                        "sed words", model.sedwModel!)["description"]
                    : null),
            icon: const Icon(Icons.arrow_drop_down_rounded),
            items: model.getModels("sed words").map<DropdownMenuItem<String>>(
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
                  model.lastInputString = null;
                },
              );
            },
          ),
          const SizedBox(height: 16),
          DropdownButtonFormField<String>(
            value: model.secModel,
            isExpanded: true,
            decoration: InputDecoration(
                prefixIcon: const Icon(Icons.looks_3, color: uniBlue),
                suffixIcon: IconButton(
                  splashRadius: 16,
                  tooltip: "Clear spelling error correction model",
                  color: uniRed,
                  icon: const Icon(Icons.clear),
                  onPressed: model.secModel != null
                      ? () {
                          setState(() {
                            model.secModel = null;
                            model.lastInputString = null;
                          });
                        }
                      : null,
                ),
                labelText: "Spelling error correction model",
                helperMaxLines: 10,
                helperText: model.secModel != null
                    ? model.getModel("sec", model.secModel!)["description"]
                    : null),
            icon: const Icon(Icons.arrow_drop_down_rounded),
            items: model.getModels("sec").map<DropdownMenuItem<String>>(
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
                model.lastInputString = null;
              });
            },
          ),
          const SizedBox(height: 16),
          Card(
            margin: EdgeInsets.zero,
            color: uniDarkGray,
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                IconButton(
                  icon: Icon(!showPipelineInfo ? Icons.info : Icons.cancel),
                  color: Colors.white,
                  onPressed: () {
                    setState(() {
                      showPipelineInfo = !showPipelineInfo;
                    });
                  },
                  splashRadius: 16,
                ),
                if (showPipelineInfo) ...[
                  const SizedBox(width: 8),
                  const Flexible(
                    child: Padding(
                      padding: EdgeInsets.all(4),
                      child: Text(
                        "The models of the pipeline will be run in order such that the output of the previous model will be used as input to the next. "
                        "If you do not specify a model for a task in the pipeline this task will be skipped. Some models (e.g. tokenization repair++) "
                        "can perform multiple tasks in one by design (e.g. correcting whitespace and spelling errors) "
                        "which is why its recommended to use them without any additional models in the pipeline that perform the same task.",
                        style: TextStyle(color: Colors.white),
                      ),
                    ),
                  ),
                ],
              ],
            ),
          ),
          const SizedBox(height: 8),
          Row(
            mainAxisAlignment: MainAxisAlignment.end,
            children: [
              ElevatedButton.icon(
                onPressed: model.validPipeline
                    ? () async {
                        await model.savePipeline();
                        setState(() {
                          showMessage(context,
                              Message("Saved pipeline settings", Status.info));
                        });
                      }
                    : null,
                icon: const Icon(Icons.save),
                label: const Text("Save pipeline settings"),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget buildInputOutput(HomeModel model) {
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Row(
              children: const [
                Expanded(
                  child: ListTile(
                    contentPadding: EdgeInsets.zero,
                    visualDensity: VisualDensity.compact,
                    title: Text(
                      "Input",
                      style: TextStyle(fontSize: 20),
                    ),
                    subtitle: Text("Enter some text to be spell checked below"),
                  ),
                ),
                SizedBox(width: 64, height: 64),
                Expanded(
                  child: ListTile(
                    contentPadding: EdgeInsets.zero,
                    visualDensity: VisualDensity.compact,
                    title: Text(
                      "Output",
                      style: TextStyle(fontSize: 20),
                    ),
                    subtitle: Text("The spell checked text will appear here"),
                  ),
                ),
              ],
            ),
            Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  child: TextField(
                    controller: model.inputController,
                    maxLines: 10,
                    maxLength: model.live ? HomeModel.maxLiveLength : null,
                    readOnly: !model.live && model.waiting,
                    maxLengthEnforcement: MaxLengthEnforcement.enforced,
                    decoration: InputDecoration(
                      border: const OutlineInputBorder(),
                      hintText: model.live
                          ? "Enter some text to spell check while typing..."
                          : "Enter some text to spell check...",
                      helperText: helperTextFromTextController(inputController),
                      helperMaxLines: 10,
                      suffixIcon: Column(
                        children: [
                          IconButton(
                            icon: const Icon(Icons.clear),
                            tooltip: "Clear text",
                            splashRadius: 16,
                            color: uniRed,
                            onPressed: !model.waiting && model.hasInput
                                ? () {
                                    setState(() {
                                      model.lastInputString = null;
                                      model.inputController.value =
                                          const TextEditingValue(
                                              text: "",
                                              selection:
                                                  TextSelection.collapsed(
                                                      offset: 0));
                                    });
                                  }
                                : null,
                          ),
                          IconButton(
                            icon: const Icon(Icons.paste),
                            tooltip: "Paste text from clipboard",
                            splashRadius: 16,
                            onPressed: !model.waiting
                                ? () async {
                                    final data =
                                        await Clipboard.getData("text/plain");
                                    setState(
                                      () {
                                        if (data != null) {
                                          final truncatedText =
                                              model.truncate(data.text!);
                                          model.inputController.value =
                                              TextEditingValue(
                                            text: truncatedText,
                                            selection: TextSelection.collapsed(
                                                offset: truncatedText.length),
                                          );
                                        } else {
                                          showMessage(
                                            context,
                                            Message(
                                                "could not paste text from clipboard",
                                                Status.error),
                                          );
                                        }
                                      },
                                    );
                                  }
                                : null,
                          ),
                          IconButton(
                            onPressed: !model.waiting
                                ? () async {
                                    try {
                                      final files = await FilePicker.platform
                                          .pickFiles(
                                              dialogTitle: "Pick a text file",
                                              type: FileType.custom,
                                              allowedExtensions: ["txt"]);
                                      if (files != null) {
                                        final file = files.files.single;
                                        final content =
                                            utf8.decode(file.bytes!);
                                        setState(() {
                                          final truncatedText =
                                              model.truncate(content);
                                          model.inputController.value =
                                              TextEditingValue(
                                            text: truncatedText,
                                            selection: TextSelection.collapsed(
                                                offset: truncatedText.length),
                                          );
                                        });
                                      }
                                    } on FormatException catch (_) {
                                      showMessage(
                                          context,
                                          Message(
                                              "error decoding file, make sure that it contains valid utf8 bytes",
                                              Status.error));
                                    } on PlatformException catch (e) {
                                      showMessage(
                                          context,
                                          Message(
                                              "error uploading file: ${e.message}",
                                              Status.error));
                                    }
                                  }
                                : null,
                            icon: const Icon(Icons.upload_file),
                            tooltip: "Upload a text file",
                            splashRadius: 16,
                          ),
                          IconButton(
                            onPressed: !model.waiting
                                ? () async {
                                    final example = await showExamplesDialog(
                                        model.examples);
                                    if (example != null) {
                                      setState(
                                        () {
                                          inputController.value =
                                              TextEditingValue(
                                                  text: example,
                                                  composing:
                                                      TextRange.collapsed(
                                                          example.length));
                                        },
                                      );
                                    }
                                  }
                                : null,
                            icon: const Icon(Icons.list),
                            tooltip: "Choose an example",
                            splashRadius: 16,
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
                SizedBox(
                  width: 64,
                  child: Column(
                    mainAxisSize: MainAxisSize.max,
                    children: [
                      Card(
                        color: model.validPipeline &&
                                !model.waiting &&
                                !model.live &&
                                inputController.text.isNotEmpty
                            ? uniBlue
                            : uniGray,
                        clipBehavior: Clip.antiAlias,
                        shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(100)),
                        child: IconButton(
                            onPressed: model.validPipeline &&
                                    !model.waiting &&
                                    !model.live &&
                                    inputController.text.isNotEmpty
                                ? () async {
                                    setState(() {
                                      outputController.value =
                                          const TextEditingValue();
                                    });
                                    final error = await model.runPipeline(
                                        model.inputController.text);
                                    if (error != null && mounted) {
                                      showMessage(context, error);
                                    }
                                  }
                                : null,
                            icon: Icon(Icons.start,
                                color: model.validPipeline &&
                                        !model.waiting &&
                                        !model.live &&
                                        inputController.text.isNotEmpty
                                    ? Colors.white
                                    : uniDarkGray),
                            tooltip: "Run pipeline on text"),
                      ),
                      IconButton(
                        icon: const Icon(Icons.stream),
                        color: model.live ? uniBlue : uniDarkGray,
                        tooltip:
                            "${!model.live ? "Enable" : "Disable"} live spell checking",
                        splashRadius: 16,
                        onPressed: model.validPipeline && !model.waiting
                            ? () async {
                                setState(
                                  () {
                                    model.live = !model.live;
                                  },
                                );
                                if (model.live) {
                                  model.lastInputString = null;
                                  final truncatedText = model
                                      .truncate(model.inputController.text);
                                  model.inputController.value =
                                      TextEditingValue(
                                    text: truncatedText,
                                    selection: TextSelection.collapsed(
                                        offset: truncatedText.length),
                                  );
                                  await model.runPipelineLive();
                                }
                              }
                            : null,
                      ),
                    ],
                  ),
                ),
                Expanded(
                  child: TextField(
                    controller: model.outputController,
                    maxLines: 10,
                    readOnly: true,
                    decoration: InputDecoration(
                      helperText: model.hasResults
                          ? "${formatS(model.runtimes["total"]["s"])}, ${formatB(model.runtimes["total"]["bps"])}/s"
                          : "",
                      helperMaxLines: 10,
                      suffixIcon: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          if (!model.waiting || model.live) ...[
                            IconButton(
                              splashRadius: 16,
                              tooltip: "Copy output to clipboard",
                              onPressed: () async {
                                await Clipboard.setData(ClipboardData(
                                    text: model.outputController.text));
                                if (mounted) {
                                  showMessage(
                                      context,
                                      Message("Copied outputs to clipboard",
                                          Status.info));
                                }
                              },
                              icon: const Icon(Icons.copy),
                            ),
                            IconButton(
                              splashRadius: 16,
                              tooltip: "Download output",
                              onPressed: () async {
                                final fileName = await FileDownloader()
                                    .downloadFile(
                                        model.outputController.text, "output");
                                if (mounted) {
                                  if (fileName != null) {
                                    showMessage(
                                      context,
                                      Message("Downloaded outputs to $fileName",
                                          Status.info),
                                    );
                                  } else {
                                    showMessage(
                                        context,
                                        Message(
                                            "Unexpected error downloading outputs",
                                            Status.error));
                                  }
                                }
                              },
                              icon: const Icon(Icons.download),
                            )
                          ] else
                            const SizedBox(
                              height: 16,
                              width: 16,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                              ),
                            )
                        ],
                      ),
                    ),
                  ),
                )
              ],
            )
          ],
        ),
      ),
    );
  }

  Widget buildEvaluation(HomeModel model) {
    return Card(
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      elevation: 2,
      clipBehavior: Clip.antiAlias,
      child: ExpansionTile(
        maintainState: true,
        title: const Text(
          "Evaluation",
          style: TextStyle(fontSize: 20),
        ),
        subtitle: const Text(
            "Evaluate the outputs of all pipeline steps against their groundtruths"),
        controlAffinity: ListTileControlAffinity.leading,
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (model.outputs.containsKey("tokenization repair"))
                Flexible(
                  child: Padding(
                    padding:
                        const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
                    child: EvaluationView(
                        inputLength: model.input.length,
                        onEvaluation: (groundtruth) async {
                          final result = await model.evaluateTr(groundtruth);
                          final error = errorMessageFromAPIResult(
                              result, "error evaluating tokenization repair");
                          if (error == null) {
                            return trEvaluationTable(result.value);
                          } else if (mounted) {
                            showMessage(context, error);
                          }
                        },
                        taskName: "tokenization repair"),
                  ),
                ),
              if (model.outputs.containsKey("sed words"))
                Flexible(
                  child: Padding(
                    padding:
                        const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
                    child: EvaluationView(
                        inputLength: model.input.length,
                        onEvaluation: (groundtruth) async {
                          final result = await model.evaluateSedw(groundtruth);
                          final error = errorMessageFromAPIResult(result,
                              "error evaluating word-level spelling error detection");
                          if (error == null) {
                            return sedwEvaluationTable(result.value);
                          } else if (mounted) {
                            showMessage(context, error);
                          }
                        },
                        taskName: "word-level spelling error detection"),
                  ),
                ),
              if (model.outputs.containsKey("sec"))
                Flexible(
                  child: Padding(
                    padding:
                        const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
                    child: EvaluationView(
                        inputLength: model.input.length,
                        onEvaluation: (groundtruth) async {
                          final result = await model.evaluateSec(groundtruth);
                          final error = errorMessageFromAPIResult(result,
                              "error evaluating spelling error correction");
                          if (error == null) {
                            return secEvaluationTable(result.value);
                          } else if (mounted) {
                            showMessage(context, error);
                          }
                        },
                        taskName: "spelling error correction"),
                  ),
                )
            ],
          ),
        ],
      ),
    );
  }

  Widget trEvaluationTable(dynamic evaluation) {
    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
        child: Table(
          defaultVerticalAlignment: TableCellVerticalAlignment.middle,
          children: [
            const TableRow(children: [
              Text(
                "Metric",
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              Text(
                "Score",
                style: TextStyle(fontWeight: FontWeight.bold),
              )
            ]),
            TableRow(children: [
              const Text("F1"),
              Text((100 * evaluation["f1"]).toStringAsFixed(2))
            ]),
            TableRow(children: [
              const Text("Precision"),
              Text((100 * evaluation["prec"]).toStringAsFixed(2))
            ]),
            TableRow(children: [
              const Text("Recall"),
              Text((100 * evaluation["rec"]).toStringAsFixed(2))
            ]),
            TableRow(children: [
              const Text("Sequence accuracy"),
              Text((100 * evaluation["seq_acc"]).toStringAsFixed(2))
            ]),
          ],
        ),
      ),
    );
  }

  Widget sedwEvaluationTable(dynamic evaluation) {
    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
        child: Table(
          defaultVerticalAlignment: TableCellVerticalAlignment.middle,
          children: [
            const TableRow(children: [
              Text(
                "Metric",
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              Text(
                "Score",
                style: TextStyle(fontWeight: FontWeight.bold),
              )
            ]),
            TableRow(children: [
              const Text("F1"),
              Text((100 * evaluation["f1"]).toStringAsFixed(2))
            ]),
            TableRow(children: [
              const Text("Precision"),
              Text((100 * evaluation["prec"]).toStringAsFixed(2))
            ]),
            TableRow(children: [
              const Text("Recall"),
              Text((100 * evaluation["rec"]).toStringAsFixed(2))
            ]),
            TableRow(children: [
              const Text("Word accuracy"),
              Text((100 * evaluation["word_acc"]).toStringAsFixed(2))
            ]),
            TableRow(children: [
              const Text("Real-word error detection rate"),
              Text(
                  "${(100 * evaluation["rw_detections"] / evaluation["rw_errors"]).toStringAsFixed(2)} (${evaluation["rw_detections"]}/${evaluation["rw_errors"]})")
            ]),
            TableRow(children: [
              const Text("Nonword error detection rate"),
              Text(
                  "${(100 * evaluation["nw_detections"] / evaluation["nw_errors"]).toStringAsFixed(2)} (${evaluation["nw_detections"]}/${evaluation["nw_errors"]})")
            ]),
          ],
        ),
      ),
    );
  }

  Widget secEvaluationTable(dynamic evaluation) {
    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
        child: Table(
          defaultVerticalAlignment: TableCellVerticalAlignment.middle,
          children: [
            const TableRow(children: [
              Text(
                "Metric",
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              Text(
                "Score",
                style: TextStyle(fontWeight: FontWeight.bold),
              )
            ]),
            TableRow(children: [
              const Text("F1"),
              Text((100 * evaluation["f1"]).toStringAsFixed(2))
            ]),
            TableRow(children: [
              const Text("Precision"),
              Text((100 * evaluation["prec"]).toStringAsFixed(2))
            ]),
            TableRow(children: [
              const Text("Recall"),
              Text((100 * evaluation["rec"]).toStringAsFixed(2))
            ]),
            TableRow(children: [
              const Text("MNED"),
              Text((evaluation["mned"]).toStringAsFixed(4))
            ]),
          ],
        ),
      ),
    );
  }

  showInfoDialog(dynamic info) async {
    if (info == null) {
      showMessage(context, Message("could not fetch info", Status.warn));
      return;
    }
    await showDialog(
      context: context,
      builder: (infoContext) {
        List<dynamic> gpus = info["gpu"];
        return SimpleDialog(
          title: const Text("Info"),
          children: [
            if (info == null) ...[
              const SimpleDialogOption(child: Text("Info not available"))
            ] else ...[
              SimpleDialogOption(
                  child: Text("Library: nsc (version ${info["version"]})")),
              SimpleDialogOption(
                  child: Text("Timeout: ${info["timeout"]} seconds")),
              SimpleDialogOption(
                  child: Text("Precision: ${info["precision"]}")),
              SimpleDialogOption(child: Text("CPU: ${info["cpu"]}")),
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
                                    padding: const EdgeInsets.all(8),
                                    child: Text("GPU $idx: $gpu"),
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
  }

  Widget exampleGroup(
      String groupName, List<String> items, Function(String) onSelected) {
    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            ListTile(
              visualDensity: VisualDensity.compact,
              title: Text(
                groupName,
                style:
                    const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
              ),
            ),
            ListView.builder(
              itemCount: items.length,
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              itemBuilder: (listContext, idx) {
                return ListTile(
                  visualDensity: VisualDensity.compact,
                  title: Text(items[idx]),
                  onTap: () {
                    onSelected(items[idx]);
                  },
                  leading: const Icon(Icons.notes),
                );
              },
            )
          ],
        ),
      ),
    );
  }

  Future<String?> showExamplesDialog(dynamic examples) async {
    if (examples == null) {
      showMessage(context, Message("could not fetch examples", Status.warn));
      return null;
    }
    return await showDialog<String?>(
      context: context,
      builder: (dialogContext) {
        final exampleGroups = examples.entries
            .map(
              (entry) {
                entry.value.sort();
                return exampleGroup(
                  entry.key,
                  entry.value.cast<String>(),
                  (item) {
                    Navigator.of(dialogContext).pop(item);
                  },
                );
              },
            )
            .toList()
            .cast<Widget>();
        return Dialog(
          child: SingleChildScrollView(
            child: Padding(
              padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 16),
              child: Column(
                children: exampleGroups
              ),
            ),
          ),
        );
      },
    );
  }
}
