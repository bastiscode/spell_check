import 'dart:convert';
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:webapp/api.dart';

import 'package:webapp/colors.dart';
import 'package:webapp/components/file_upload.dart';
import 'package:webapp/components/input_or_upload.dart';
import 'package:webapp/components/message.dart';
import 'package:webapp/utils.dart';

import 'package:webapp/platforms/file_download_interface.dart';

typedef OnDownload = void Function(String?, String);
typedef OnClipboard = void Function(String);

class EvaluationView extends StatefulWidget {
  final List<String> input;
  final dynamic output;
  final dynamic runtimes;
  final OnDownload onDownload;
  final OnClipboard onClipboard;

  const EvaluationView(
  {required this.input,
  required this.output,
  required this.runtimes,
  required this.onDownload,
  required this.onClipboard,
  super.key});

@override
State<EvaluationView> createState() => _EvaluationViewState();
}

class _EvaluationViewState extends State<EvaluationView> {
  String? trGroundtruth;
  String? sedwGroundtruth;
  String? secGroundtruth;

  dynamic trEvaluation;
  dynamic sedwEvaluation;
  dynamic secEvaluation;

  @override
  Widget build(BuildContext context) {
    // setup data
    bool hasTr = false;
    bool hasSedw = false;
    bool hasSec = false;

    List<String> trInput = widget.input;
    List<String> sedwInput = widget.input;
    List<String> secInput = widget.input;

    dynamic trResults;
    dynamic sedwResults;
    dynamic secResults;

    dynamic trRuntimes;
    dynamic sedwRuntimes;
    dynamic secRuntimes;

    if (widget.output.containsKey("tokenization repair")) {
      hasTr = true;
      trResults = widget.output["tokenization repair"];
      trRuntimes = widget.runtimes["tokenization repair"];
      sedwInput = trResults["text"].cast<String>();
      secInput = trResults["text"].cast<String>();
    }
    if (widget.output.containsKey("sed words")) {
      hasSedw = true;
      sedwResults = widget.output["sed words"];
      sedwRuntimes = widget.runtimes["sed words"];
      secInput = sedwResults["text"].cast<String>();
      for (final detection in sedwResults["detections"]) {
        _rawDetections.add(detection.join(" "));
      }
    }
    if (widget.output.containsKey("sec")) {
      hasSec = true;
      secResults = widget.output["sec"];
      secRuntimes = widget.runtimes["sec"];
    }

    return Row(
          crossAxisAlignment: CrossAxisAlignment.center,
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            if (hasTr)
              Flexible(
                child: ListTile(
                  title: const Text(
                    "Tokenization repair",
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                  subtitle: Text(
                      "${formatS(trRuntimes["s"])}, ${formatB(trRuntimes["bps"])}/s"),
                  trailing: resultActions(
                      "tokenization repair",
                      trResults["text"].join("\n") + "\n",
                      "tr_results",
                          () {
                        widget.onClipboard("tokenization repair");
                      },
                          (fileName) {
                        widget.onDownload(fileName, "tokenization repair");
                      }),
                ),
              ),
            if (hasSedw) ...[
              const Icon(Icons.arrow_right, color: Colors.transparent,), // just for spacing
              Flexible(
                child: ExpansionTile(
                  tilePadding: EdgeInsets.zero,
                  controlAffinity: ListTileControlAffinity.leading,
                  title: const Text("Word-level spelling error detection",
                      style: TextStyle(fontWeight: FontWeight.bold)),
                  subtitle: Text(
                      "${formatS(sedwRuntimes["s"])}, ${formatB(sedwRuntimes["bps"])}/s"),
                  trailing: resultActions(
                      "spelling error detection",
                      "${_rawDetections.join("\n")}\n",
                      "sedw_results",
                          () {
                        widget.onClipboard("spelling error detection");
                      },
                          (fileName) {
                        widget.onDownload(fileName, "spelling error detection");
                      }),
                  children: [
                    const Text(
                      "Evaluate spelling error detection results on groundtruths",
                      style: TextStyle(fontWeight: FontWeight.bold),
                    ),
                    sedwGroundtruth == null
                        ? ElevatedButton.icon(
                        onPressed: () async {
                          sedwGroundtruth = await showInputOrUploadDialog(
                              context,
                              "Set tokenization repair groundtruth");
                          setState(() {});
                        },
                        label: const Text(
                            "Set tokenization repair groundtruth"),
                        icon: const Icon(Icons.text_snippet))
                        : Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Expanded(
                          child: Uploaded(
                            title: "Got tokenization repair groundtruth",
                            name: "Tokenization repair groundtruth",
                            bytes: utf8.encode(sedwGroundtruth!).length,
                            lines: sedwGroundtruth!.split("\n").length,
                            onDelete: () {
                              setState(() {
                                sedwGroundtruth = null;
                                sedwEvaluation = null;
                              });
                            },
                          ),
                        ),
                        const SizedBox(width: 8),
                        Flexible(
                          child: ElevatedButton.icon(
                            onPressed: () async {
                              final result = await api.evaluateTr(
                                  trInput.join("\n"),
                                  trResults["text"].join("\n"),
                                  trGroundtruth!);
                              final error = errorMessageFromAPIResult(
                                  result, "error during evaluation");
                              if (error != null) {
                                if (mounted) {
                                  showMessage(context, error);
                                }
                              } else {
                                setState(() {
                                  trEvaluation = result.value;
                                });
                              }
                            },
                            icon: const Icon(Icons.analytics),
                            label: const Text("Evaluate"),
                          ),
                        )
                      ],
                    ),
                    if (trEvaluation != null) Text("$trEvaluation")
                  ],
                ),
              ),
            ],
            if (hasSec) ...[
              const Icon(Icons.arrow_right, color: Colors.transparent), // just for spacing
              Flexible(
                child: ExpansionTile(
                  tilePadding: EdgeInsets.zero,
                  controlAffinity: ListTileControlAffinity.leading,
                  title: const Text("Spelling error correction",
                      style: TextStyle(fontWeight: FontWeight.bold)),
                  subtitle: Text(
                      "${formatS(secRuntimes["s"])}, ${formatB(secRuntimes["bps"])}/s"),
                  trailing: resultActions(
                      "spelling error correction",
                      "${secResults["text"].join("\n")}\n",
                      "sec_results",
                          () {
                        widget.onClipboard("spelling error correction");
                      },
                          (fileName) {
                        widget.onDownload(
                            fileName, "spelling error correction");
                      }),
                  childrenPadding: const EdgeInsets.symmetric(vertical: 8),
                  children: [
                    const Text(
                      "Evaluate spelling error correction results on groundtruths",
                      style: TextStyle(fontWeight: FontWeight.bold),
                    ),
                    secGroundtruth == null
                        ? ElevatedButton.icon(
                        onPressed: () async {
                          secGroundtruth = await showInputOrUploadDialog(
                              context,
                              "Set spelling error correction groundtruth");
                          setState(() {});
                        },
                        label: const Text(
                            "Set spelling error correction groundtruth"),
                        icon: const Icon(Icons.text_snippet))
                        : Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Expanded(
                          child: Uploaded(
                            title:
                            "Got spelling error correction groundtruth",
                            name: "Spelling error correction groundtruth",
                            bytes: utf8.encode(secGroundtruth!).length,
                            lines: secGroundtruth!.split("\n").length,
                            onDelete: () {
                              setState(() {
                                secGroundtruth = null;
                                secEvaluation = null;
                              });
                            },
                          ),
                        ),
                        const SizedBox(width: 8),
                        Flexible(
                          child: ElevatedButton.icon(
                            onPressed: () async {
                              final result = await api.evaluateSec(
                                  secInput.join("\n"),
                                  secResults["text"].join("\n"),
                                  secGroundtruth!);
                              final error = errorMessageFromAPIResult(
                                  result, "error during evaluation");
                              if (error != null) {
                                if (mounted) {
                                  showMessage(context, error);
                                }
                              } else {
                                setState(() {
                                  trEvaluation = result.value;
                                });
                              }
                            },
                            icon: const Icon(Icons.analytics),
                            label: const Text("Evaluate"),
                          ),
                        )
                      ],
                    ),
                    if (secEvaluation != null) Text("$secEvaluation")
                  ],
                ),
              ),
            ],
          ],
        );
  }
}

Widget resultActions(
    String tooltipName,
    String rawOutput,
    String downloadFileName,
    VoidCallback onClipboard,
    Function(String?) onDownload) {
  return Row(
    mainAxisSize: MainAxisSize.min,
    mainAxisAlignment: MainAxisAlignment.center,
    children: [
      IconButton(
        onPressed: () async {
          await Clipboard.setData(ClipboardData(text: rawOutput));
          onClipboard();
        },
        tooltip: "Copy raw $tooltipName outputs",
        splashRadius: 16,
        icon: const Icon(Icons.content_copy),
      ),
      IconButton(
        onPressed: () async {
          final downloader = FileDownloader();
          final fileName =
          await downloader.downloadFile(rawOutput, downloadFileName);
          onDownload(fileName);
        },
        tooltip: "Download raw $tooltipName outputs",
        splashRadius: 16,
        icon: const Icon(Icons.file_download),
      )
    ],
  );
}

Widget buildDeletionsText(String s, List<dynamic> edits) {
  if (s.isEmpty) {
    return Text(s);
  }
  final chars = s.characters.toList();
  List<InlineSpan>? children = [];
  for (int idx = 1; idx < chars.length; idx += 1) {
    final char = chars[idx];
    final edit = edits[idx];
    children.add(TextSpan(
        text: char,
        style: edit == 2 ? const TextStyle(backgroundColor: uniRed) : null));
  }
  return Text.rich(TextSpan(text: chars.first, children: children));
}
