import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'package:webapp/colors.dart';

import 'package:webapp/platforms/file_download_interface.dart';

typedef OnDownload = Function(String?);

class TrResultView extends StatefulWidget {
  final List<String> input;
  final dynamic result;
  final OnDownload onDownload;

  const TrResultView(
      {required this.input,
      required this.result,
      required this.onDownload,
      super.key});

  @override
  State<TrResultView> createState() => _TrResultViewState();
}

class _TrResultViewState extends State<TrResultView> {
  bool _showRaw = false;

  @override
  Widget build(BuildContext context) {
    final List repaired = widget.result["output"]["text"];
    final List? edited = widget.result["output"]["edited"];
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Row(
          children: [
            const Text("Tokenization repair results",
                style: TextStyle(fontWeight: FontWeight.bold)),
            const Spacer(),
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
                    ClipboardData(text: "${repaired.join("\n")}\n"));
              },
              tooltip: "Copy raw outputs",
              splashRadius: 16,
              icon: const Icon(Icons.content_copy),
            ),
            IconButton(
              onPressed: () async {
                final downloader = FileDownloader();
                final fileName = await downloader.downloadFile(
                    "${repaired.join("\n")}\n", "tr_result");
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
            itemCount: widget.input.length,
            physics: const NeverScrollableScrollPhysics(),
            shrinkWrap: true,
            itemBuilder: (buildContext, idx) {
              final input = widget.input[idx];
              return Row(
                children: [
                  if (!_showRaw)
                    Flexible(
                        child: edited == null
                            ? Text(input)
                            : buildDeletionsText(input, edited[idx])),
                  if (!_showRaw) const Icon(Icons.arrow_right_alt),
                  Flexible(child: Text(repaired[idx]))
                ],
              );
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
