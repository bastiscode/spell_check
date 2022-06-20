import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'package:webapp/platforms/file_download_interface.dart';

typedef OnDownload = Function(String?);

class SecResultView extends StatefulWidget {
  final List<String> input;
  final dynamic result;
  final dynamic runtime;
  final OnDownload onDownload;

  const SecResultView(
      {required this.input,
      required this.result,
      required this.runtime,
      required this.onDownload,
      super.key});

  @override
  State<SecResultView> createState() => _SecResultViewState();
}

class _SecResultViewState extends State<SecResultView> {
  bool _showRaw = false;

  @override
  Widget build(BuildContext context) {
    final List corrected = widget.result["text"];
    final Map? edited =
        widget.result.containsKey("edited") ? widget.result["edited"] : null;
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        const Text("Spelling error correction results",
            style: TextStyle(fontWeight: FontWeight.bold)),
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
                    ClipboardData(text: "${corrected.join("\n")}\n"));
              },
              tooltip: "Copy raw outputs",
              splashRadius: 16,
              icon: const Icon(Icons.content_copy),
            ),
            IconButton(
              onPressed: () async {
                final downloader = FileDownloader();
                final fileName = await downloader.downloadFile(
                    "${corrected.join("\n")}\n", "sec_result");
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
              if (_showRaw) {
                return Text(corrected[idx]);
              } else if (edited == null) {
                return Row(
                  children: [
                    Flexible(child: Text(input)),
                    const Icon(Icons.arrow_right_alt),
                    Flexible(child: Text(corrected[idx]))
                  ],
                );
              } else {
                return Row(
                  children: [
                    Flexible(child: Text(input)),
                    const Icon(Icons.arrow_right_alt),
                    Flexible(child: Text(corrected[idx]))
                  ],
                );
              }
            },
          ),
        )
      ],
    );
  }
}
