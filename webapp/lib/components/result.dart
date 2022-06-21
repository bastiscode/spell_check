import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'package:webapp/colors.dart';
import 'package:webapp/components/message.dart';
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
  final List<String> _rawDetections = [];
  List<int> _indices = [0];
  final TextEditingController _filterController = TextEditingController();

  List<int> getIndices(String filter) {
    List<int> indices = [];
    for (var pattern in filter.split(",")) {
      pattern = pattern.trim().replaceAll(r"\s+", "");
      if (pattern == "") {
        continue;
      } else if (RegExp(r"^\d+$").hasMatch(pattern)) {
        indices.add(int.parse(pattern) - 1);
      } else if (RegExp(r"^\d+-\d+$").hasMatch(pattern)) {
        final split = pattern.split("-");
        final first = int.parse(split[0]);
        final second = int.parse(split[1]);
        for (var i = first - 1; i < second; i++) {
          indices.add(i);
        }
      } else {
        showMessage(context,
            Message("got invalid pattern '$pattern' in filter", Status.warn));
      }
    }
    if (indices.isEmpty) {
      indices.add(0);
    }
    indices =
        indices.where((idx) => idx >= 0 && idx < widget.input.length).toList();
    indices.sort();
    return indices;
  }

  @override
  Widget build(BuildContext context) {
    // setup data
    bool hasTr = false;
    bool hasSedw = false;
    bool hasSec = false;

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
    }
    if (widget.output.containsKey("sed words")) {
      hasSedw = true;
      sedwResults = widget.output["sed words"];
      sedwRuntimes = widget.runtimes["sed words"];
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
          children: [
            Expanded(
              child: TextField(
                decoration: const InputDecoration(
                    prefixIcon: Icon(Icons.filter_alt),
                    border: OutlineInputBorder(),
                    hintText:
                        "Filter results by specifying indices like 4, 5, 6, or ranges like 10-20, 40-50. By default or if no valid filter pattern is found only the first result is shown."),
                controller: _filterController,
                onSubmitted: (filter) {
                  setState(() {
                    _indices = getIndices(filter);
                  });
                },
              ),
            ),
          ],
        ),
        Row(
          crossAxisAlignment: CrossAxisAlignment.center,
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            Flexible(
              child: Card(
                child: ListTile(
                  title: const Text(
                    "Input",
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                  subtitle: Text(
                      "${widget.input.length} line${widget.input.length > 1 ? "s" : ""}"),
                ),
              ),
            ),
            if (hasTr)
              Flexible(
                flex: 1,
                child: Card(
                  child: ListTile(
                    title: const Text(
                      "Tokenization repair",
                      style: TextStyle(fontWeight: FontWeight.bold),
                    ),
                    subtitle: Text(
                        "${formatS(trRuntimes["s"])}, ${formatB(trRuntimes["bps"])}/s"),
                    trailing: resultActions("tokenization repair",
                        trResults["text"].join("\n") + "\n", "tr_results", () {
                      widget.onClipboard("tokenization repair");
                    }, (fileName) {
                      widget.onDownload(fileName, "tokenization repair");
                    }),
                  ),
                ),
              ),
            if (hasSedw)
              Flexible(
                flex: 1,
                child: Card(
                  child: ListTile(
                    title: const Text("Word-level spelling error detection",
                        style: TextStyle(fontWeight: FontWeight.bold)),
                    subtitle: Text(
                        "${formatS(sedwRuntimes["s"])}, ${formatB(sedwRuntimes["bps"])}/s"),
                    trailing: resultActions("spelling error detection",
                        "${_rawDetections.join("\n")}\n", "sedw_results", () {
                      widget.onClipboard("spelling error detection");
                    }, (fileName) {
                      widget.onDownload(fileName, "spelling error detection");
                    }),
                  ),
                ),
              ),
            if (hasSec)
              Flexible(
                flex: 1,
                child: Card(
                  child: ListTile(
                    title: const Text("Spelling error correction",
                        style: TextStyle(fontWeight: FontWeight.bold)),
                    subtitle: Text(
                        "${formatS(secRuntimes["s"])}, ${formatB(secRuntimes["bps"])}/s"),
                    trailing: resultActions(
                        "spelling error correction",
                        "${secResults["text"].join("\n")}\n",
                        "sec_results", () {
                      widget.onClipboard("spelling error correction");
                    }, (fileName) {
                      widget.onDownload(fileName, "spelling error correction");
                    }),
                  ),
                ),
              ),
          ],
        ),
        Flexible(
          child: ListView.builder(
            padding: EdgeInsets.zero,
            itemCount: _indices.length,
            physics: const NeverScrollableScrollPhysics(),
            shrinkWrap: true,
            itemBuilder: (buildContext, i) {
              final idx = _indices[i];
              List<Widget> children = [
                Flexible(
                  child: Card(
                    child: ListTile(
                      title: Text(
                        widget.input[idx],
                        textAlign: TextAlign.center,
                      ),
                      visualDensity: VisualDensity.compact,
                    ),
                  ),
                )
              ];
              if (hasTr) {
                children.add(
                  Flexible(
                    flex: 1,
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
                children.addAll([
                  Flexible(
                    flex: 1,
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
                ]);
              }
              if (hasSec) {
                children.addAll([
                  Flexible(
                    flex: 1,
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
                ]);
              }
              return Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Row(
                    children: [
                      const SizedBox(
                          width: 16,
                          child: Divider(
                            thickness: 1,
                          )),
                      const SizedBox(width: 8),
                      Text("${idx + 1}"),
                      const SizedBox(width: 8),
                      const Expanded(
                        child: Divider(
                          thickness: 1,
                        ),
                      )
                    ],
                  ),
                  Row(
                      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                      children: children),
                ],
              );
            },
          ),
        ),
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
