import 'dart:convert';
import 'dart:math';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:webapp/api.dart';

import 'package:webapp/colors.dart';
import 'package:webapp/components/file_upload.dart';
import 'package:webapp/components/input_or_upload.dart';
import 'package:webapp/utils.dart';

import 'package:webapp/platforms/file_download_interface.dart';

typedef OnDownload = void Function(String?, String);
typedef OnClipboard = void Function(String);

class ResultView extends StatefulWidget {
  final List<String> input;
  final dynamic output;
  final dynamic runtimes;
  final OnDownload onDownload;
  final OnClipboard onClipboard;

  const ResultView(
      {required this.input,
      required this.output,
      required this.runtimes,
      required this.onDownload,
      required this.onClipboard,
      super.key});

  @override
  State<ResultView> createState() => _ResultViewState();
}

class _ResultViewState extends State<ResultView> {
  bool showTrRaw = false;
  bool showSedwRaw = false;
  bool showSecRaw = false;

  String? trGroundtruth;
  String? sedwGroundtruth;
  String? secGroundtruth;

  dynamic trEvaluation;
  dynamic sedwEvaluation;
  dynamic secEvaluation;

  int numShow = 10;

  final List<String> _rawDetections = [];

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

    return Column(
      crossAxisAlignment: CrossAxisAlignment.center,
      mainAxisSize: MainAxisSize.min,
      children: [
        Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            if (hasTr)
              Flexible(
                child: ExpansionTile(
                  expandedCrossAxisAlignment: CrossAxisAlignment.start,
                  tilePadding: EdgeInsets.zero,
                  title: const Text(
                    "Tokenization repair results",
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                  controlAffinity: ListTileControlAffinity.leading,
                  subtitle: Text(
                      "${formatS(trRuntimes["s"])}, ${formatB(trRuntimes["bps"])}/s"),
                  trailing: resultActions(
                      "tokenization repair",
                      showTrRaw,
                      () {
                        setState(() {
                          showTrRaw = !showTrRaw;
                        });
                      },
                      trResults["text"].join("\n") + "\n",
                      "tr_results",
                      () {
                        widget.onClipboard("tokenization repair");
                      },
                      (fileName) {
                        widget.onDownload(fileName, "tokenization repair");
                      },
                      () {}),
                  childrenPadding: const EdgeInsets.symmetric(vertical: 8),
                  children: [
                    trGroundtruth == null
                        ? ElevatedButton.icon(
                            onPressed: () async {
                              trGroundtruth = await showInputOrUploadDialog(
                                  context,
                                  "Get tokenization repair groundtruth");
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
                                  title: "Got groundtruth",
                                  name: "Tokenization repair groundtruth",
                                  bytes: utf8.encode(trGroundtruth!).length,
                                  lines: trGroundtruth!.split("\n").length,
                                  onDelete: () {
                                    setState(() {
                                      trGroundtruth = null;
                                    });
                                  },
                                ),
                              ),
                              const SizedBox(width: 8),
                              Flexible(
                                child: ElevatedButton.icon(
                                  onPressed: () {},
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
            if (hasSedw)
              Flexible(
                child: ExpansionTile(
                    tilePadding: EdgeInsets.zero,
                    controlAffinity: ListTileControlAffinity.leading,
                    title: const Text(
                        "Word-level spelling error detection results",
                        style: TextStyle(fontWeight: FontWeight.bold)),
                    subtitle: Text(
                        "${formatS(sedwRuntimes["s"])}, ${formatB(sedwRuntimes["bps"])}/s"),
                    trailing: resultActions(
                        "spelling error detection",
                        showSedwRaw,
                        () {
                          setState(() {
                            showSedwRaw = !showSedwRaw;
                          });
                        },
                        "${_rawDetections.join("\n")}\n",
                        "sedw_results",
                        () {
                          widget.onClipboard("spelling error detection");
                        },
                        (fileName) {
                          widget.onDownload(
                              fileName, "spelling error detection");
                        },
                        () {})),
              ),
            if (hasSec)
              Flexible(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    Text(
                      "Spelling error correction results (${formatS(secRuntimes["s"])}, ${formatB(secRuntimes["bps"])}/s)",
                      style: const TextStyle(fontWeight: FontWeight.bold),
                      textAlign: TextAlign.center,
                    ),
                    resultActions(
                        "spelling error correction",
                        showSecRaw,
                        () {
                          setState(() {
                            showSecRaw = !showSecRaw;
                          });
                        },
                        secResults["text"].join("\n") + "\n",
                        "sec_results",
                        () {
                          widget.onClipboard("spelling error correction");
                        },
                        (fileName) {
                          widget.onDownload(
                              fileName, "spelling error correction");
                        },
                        () {})
                  ],
                ),
              ),
          ],
        ),
        Flexible(
          child: ListView.builder(
            padding: EdgeInsets.zero,
            itemCount: min(numShow, widget.input.length),
            physics: const NeverScrollableScrollPhysics(),
            shrinkWrap: true,
            itemBuilder: (buildContext, idx) {
              List<Widget> children = [];
              if (hasTr) {
                children.add(
                  Flexible(
                    child: Card(
                      child: ListTile(
                        title: Text(
                          trResults["text"][idx],
                          textAlign: TextAlign.center,
                        ),
                        visualDensity: VisualDensity.compact,
                      ),
                    ),
                  ),
                );
              }
              if (hasSedw) {
                children.add(
                  Flexible(
                    child: Card(
                      child: ListTile(
                        title: Text(
                          sedwResults["text"][idx],
                          textAlign: TextAlign.center,
                        ),
                        visualDensity: VisualDensity.compact,
                      ),
                    ),
                  ),
                );
              }
              if (hasSec) {
                children.add(
                  Flexible(
                    child: Card(
                      child: ListTile(
                        title: Text(
                          secResults["text"][idx],
                          textAlign: TextAlign.center,
                        ),
                        visualDensity: VisualDensity.compact,
                      ),
                    ),
                  ),
                );
              }
              return Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: children);
            },
          ),
        ),
        if (widget.input.length > numShow)
          Flexible(
            child: ListTile(
              title: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  if (widget.input.length > numShow + 10) ...[
                    ElevatedButton.icon(
                      onPressed: numShow < widget.input.length
                          ? () {
                              setState(() {
                                numShow = numShow + 10;
                              });
                            }
                          : null,
                      icon: const Icon(Icons.expand_more),
                      label: const Text("Show 10 more results"),
                    ),
                    const SizedBox(
                      width: 8,
                    ),
                  ],
                  ElevatedButton.icon(
                    onPressed: numShow < widget.input.length
                        ? () {
                            setState(() {
                              numShow = widget.input.length;
                            });
                          }
                        : null,
                    icon: const Icon(Icons.list),
                    label: Text("Show all ${widget.input.length} results"),
                  )
                ],
              ),
            ),
          )
      ],
    );
  }
}

Widget resultActions(
    String tooltipName,
    bool showRaw,
    VoidCallback onRawChanged,
    String rawOutput,
    String downloadFileName,
    VoidCallback onClipboard,
    Function(String?) onDownload,
    VoidCallback onEvaluate) {
  return Row(
    mainAxisSize: MainAxisSize.min,
    mainAxisAlignment: MainAxisAlignment.center,
    children: [
      IconButton(
        onPressed: onRawChanged,
        tooltip: "Toggle between raw and formatted $tooltipName output",
        splashRadius: 16,
        icon: Icon(showRaw ? Icons.raw_on : Icons.raw_off),
      ),
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
      ),
      IconButton(
          onPressed: onEvaluate,
          tooltip: "Evaluate $tooltipName outputs",
          splashRadius: 16,
          icon: const Icon(Icons.analytics))
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
