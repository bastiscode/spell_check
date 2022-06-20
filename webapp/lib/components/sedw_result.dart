import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'package:webapp/colors.dart';

import 'package:webapp/platforms/file_download_interface.dart';

typedef OnDownload = Function(String?);

class SedwResultView extends StatefulWidget {
  final List<String> input;
  final dynamic result;
  final dynamic runtime;
  final OnDownload onDownload;

  const SedwResultView(
      {required this.input,
      required this.result,
  required this.runtime,
      required this.onDownload,
      super.key});

  @override
  State<SedwResultView> createState() => _SedwResultViewState();
}

class _SedwResultViewState extends State<SedwResultView> {
  bool _showRaw = false;
  final List<String> _rawDetections = [];

  @override
  void initState() {
    super.initState();
    for (final detection in widget.result["detections"]) {
      _rawDetections.add(detection.join(" "));
    }
  }

  @override
  Widget build(BuildContext context) {
    final List text = widget.result["text"];
    final List detections = widget.result["detections"];
    return Column(
      crossAxisAlignment: CrossAxisAlignment.center,
      mainAxisSize: MainAxisSize.min,
      children: [
        const Text("Word-level spelling error detection results",
            style: TextStyle(fontWeight: FontWeight.bold),
        textAlign: TextAlign.center,),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            IconButton(
              onPressed: () {
                setState(() {
                  _showRaw = !_showRaw;
                });
              },
              tooltip: "Toggle between raw and formatted output",
              splashRadius: 16,
              icon: Icon(_showRaw ? Icons.raw_on : Icons.raw_off),
            ),
            IconButton(
              onPressed: () {
                Clipboard.setData(
                    ClipboardData(text: "${_rawDetections.join("\n")}\n"));
              },
              tooltip: "Copy raw outputs",
              splashRadius: 16,
              icon: const Icon(Icons.content_copy),
            ),
            IconButton(
              onPressed: () async {
                final downloader = FileDownloader();
                final fileName = await downloader.downloadFile(
                    "${_rawDetections.join("\n")}\n", "sedw_result");
                widget.onDownload(fileName);
              },
              tooltip: "Download raw outputs",
              splashRadius: 16,
              icon: const Icon(Icons.file_download),
            ),
          ],
        ),
        Flexible(
          child: ListView.builder(
            itemCount: text.length,
            physics: const NeverScrollableScrollPhysics(),
            shrinkWrap: true,
            itemBuilder: (buildContext, idx) {
              List<Widget> children = [];
              if (_showRaw) {
                children.add(Text(_rawDetections[idx]));
              } else {
                children.addAll([]);
              }
              return Row(children: children);
            },
          ),
        )
      ],
    );
  }
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
