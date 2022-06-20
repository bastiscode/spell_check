import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'package:webapp/colors.dart';

import 'package:webapp/platforms/file_download_interface.dart';

typedef OnDownload = void Function(String?, String);

class ResultView extends StatefulWidget {
  final List<String> input;
  final dynamic output;
  final dynamic runtimes;
  final OnDownload onDownload;

  const ResultView(
      {required this.input,
      required this.output,
      required this.runtimes,
      required this.onDownload,
      super.key});

  @override
  State<ResultView> createState() => _ResultViewState();
}

class _ResultViewState extends State<ResultView> {
  bool showTrRaw = false;
  bool showSedwRaw = false;
  bool showSecRaw = false;

  bool hasTr = false;
  bool hasSedw = false;
  bool hasSec = false;

  late final List<String> trInput;
  late List<String> sedwInput;
  late List<String> secInput;

  late final dynamic trResults;
  late final dynamic sedwResults;
  late final dynamic secResults;

  late final dynamic trRuntimes;
  late final dynamic sedwRuntimes;
  late final dynamic secRuntimes;

  final List<String> _rawDetections = [];

  @override
  void initState() {
    super.initState();

    trInput = widget.input;
    sedwInput = widget.input;
    secInput = widget.input;

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
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.center,
      mainAxisSize: MainAxisSize.min,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            if (hasTr)
              Flexible(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    const Text(
                      "Tokenization repair results",
                      style: TextStyle(fontWeight: FontWeight.bold),
                      textAlign: TextAlign.center,
                    ),
                    resultActions(
                      showTrRaw,
                      () {
                        setState(() {
                          showTrRaw = !showTrRaw;
                        });
                      },
                      trResults["text"].join("\n") + "\n",
                      "tr_results",
                      (fileName) {
                        widget.onDownload(
                            fileName, "tokenization repair outputs");
                      },
                    )
                  ],
                ),
              ),
            if (hasSedw)
              Flexible(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    const Text(
                      "Word-level spelling error detection results",
                      style: TextStyle(fontWeight: FontWeight.bold),
                      textAlign: TextAlign.center,
                    ),
                    resultActions(
                      showSedwRaw,
                      () {
                        setState(() {
                          showSedwRaw = !showSedwRaw;
                        });
                      },
                      "${_rawDetections.join("\n")}\n",
                      "sedw_results",
                      (fileName) {
                        widget.onDownload(
                            fileName, "spelling error detection outputs");
                      },
                    )
                  ],
                ),
              ),
            if (hasSec)
              Flexible(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    const Text(
                      "Spelling error correction results",
                      style: TextStyle(fontWeight: FontWeight.bold),
                      textAlign: TextAlign.center,
                    ),
                    resultActions(
                      showSecRaw,
                      () {
                        setState(() {
                          showSecRaw = !showSecRaw;
                        });
                      },
                      secResults["text"].join("\n") + "\n",
                      "sec_results",
                      (fileName) {
                        widget.onDownload(
                            fileName, "spelling error correction outputs");
                      },
                    )
                  ],
                ),
              ),
          ],
        ),
        Flexible(
          child: ListView.builder(
            padding: EdgeInsets.zero,
            itemCount: widget.input.length,
            physics: const NeverScrollableScrollPhysics(),
            shrinkWrap: true,
            itemBuilder: (buildContext, idx) {
              List<Widget> children = [];
              if (hasTr) {
                children.add(Flexible(
                    child: Card(
                        child: ListTile(
                  title: Text(
                    trResults["text"][idx],
                    textAlign: TextAlign.center,
                  ),
                  visualDensity: VisualDensity.compact,
                ))));
              }
              if (hasSedw) {
                children.add(Flexible(
                    child: Card(
                        child: ListTile(
                  title: Text(
                    sedwResults["text"][idx],
                    textAlign: TextAlign.center,
                  ),
                  visualDensity: VisualDensity.compact,
                ))));
              }
              if (hasSec) {
                children.add(Flexible(
                    child: Card(
                        child: ListTile(
                  title: Text(
                    secResults["text"][idx],
                    textAlign: TextAlign.center,
                  ),
                  visualDensity: VisualDensity.compact,
                ))));
              }
              return Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: children);
            },
          ),
        )
      ],
    );
  }
}

Widget resultActions(bool showRaw, VoidCallback onRawChanged, String rawOutput,
    String downloadFileName, Function(String?) onDownload) {
  return Row(
    mainAxisAlignment: MainAxisAlignment.center,
    children: [
      IconButton(
        onPressed: onRawChanged,
        tooltip: "Toggle between raw and formatted output",
        splashRadius: 16,
        icon: Icon(showRaw ? Icons.raw_on : Icons.raw_off),
      ),
      IconButton(
        onPressed: () {
          Clipboard.setData(ClipboardData(text: rawOutput));
        },
        tooltip: "Copy raw outputs",
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
        tooltip: "Download raw outputs",
        splashRadius: 16,
        icon: const Icon(Icons.file_download),
      ),
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
