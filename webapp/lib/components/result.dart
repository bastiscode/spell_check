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

  const ResultView(
      {required this.input,
      required this.output,
      required this.runtimes,
      super.key});

  @override
  State<ResultView> createState() => _ResultViewState();
}

class _ResultViewState extends State<ResultView> {
  List<int> _indices = List.generate(10, (i) => i);
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
      indices.addAll(List.generate(10, (i) => i));
    }
    return indices;
  }

  @override
  Widget build(BuildContext context) {
    // setup data
    bool hasTr = widget.output.containsKey("whitespace correction");
    bool hasSedw = widget.output.containsKey("sed words");
    bool hasSec = widget.output.containsKey("sec");

    dynamic trResults;
    dynamic sedwResults;
    dynamic secResults;

    dynamic trRuntimes;
    dynamic sedwRuntimes;
    dynamic secRuntimes;

    if (hasTr) {
      trResults = widget.output["whitespace correction"];
      trRuntimes = widget.runtimes["whitespace correction"];
    }
    if (hasSedw) {
      sedwResults = widget.output["sed words"];
      sedwRuntimes = widget.runtimes["sed words"];
    }
    if (hasSec) {
      secResults = widget.output["sec"];
      secRuntimes = widget.runtimes["sec"];
    }

    final indices =
        _indices.where((idx) => idx >= 0 && idx < widget.input.length).toList();
    indices.sort();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.center,
      mainAxisSize: MainAxisSize.min,
      children: [
        if (widget.input.length > 1)
          Row(
            children: [
              Expanded(
                child: Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  child: TextField(
                    decoration: const InputDecoration(
                        prefixIcon: Icon(Icons.filter_alt),
                        border: OutlineInputBorder(),
                        hintText:
                            "Filter results by specifying indices like 4, 5, 6, or ranges like 10-20, 40-50. "
                            "By default or if no valid filter pattern is found the first 10 results are shown."),
                    controller: _filterController,
                    onSubmitted: (filter) {
                      setState(() {
                        _indices = getIndices(filter);
                      });
                    },
                  ),
                ),
              ),
            ],
          ),
        Row(
          crossAxisAlignment: CrossAxisAlignment.center,
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            Flexible(
              child: ListTile(
                visualDensity: VisualDensity.compact,
                title: const Text(
                  "Input",
                  style: TextStyle(fontWeight: FontWeight.bold),
                ),
                subtitle: Text(
                    "${widget.input.length} line${widget.input.length > 1 ? "s" : ""}"),
              ),
            ),
            if (hasTr)
              Flexible(
                flex: 1,
                child: ListTile(
                  visualDensity: VisualDensity.compact,
                  title: const Text(
                    "Whitespace correction",
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                  subtitle: Text(
                      "${formatS(trRuntimes["s"])}, ${formatB(trRuntimes["bps"])}/s"),
                  trailing: resultActions(context, "whitespace correction",
                      trResults["text"].join("\n"), "tr_results"),
                ),
              ),
            if (hasSedw)
              Flexible(
                flex: 1,
                child: ListTile(
                  visualDensity: VisualDensity.compact,
                  title: const Text("Word-level spelling error detection",
                      style: TextStyle(fontWeight: FontWeight.bold)),
                  subtitle: Text(
                      "${formatS(sedwRuntimes["s"])}, ${formatB(sedwRuntimes["bps"])}/s"),
                  trailing: resultActions(
                      context,
                      "spelling error detection",
                      "${sedwResults["detections"].map((detection) => detection.join(" ")).join("\n")}",
                      "sedw_results"),
                ),
              ),
            if (hasSec)
              Flexible(
                flex: 1,
                child: ListTile(
                  visualDensity: VisualDensity.compact,
                  title: const Text("Spelling error correction",
                      style: TextStyle(fontWeight: FontWeight.bold)),
                  subtitle: Text(
                      "${formatS(secRuntimes["s"])}, ${formatB(secRuntimes["bps"])}/s"),
                  trailing: resultActions(context, "spelling error correction",
                      "${secResults["text"].join("\n")}", "sec_results"),
                ),
              ),
          ],
        ),
        Flexible(
          child: ListView.builder(
            padding: EdgeInsets.zero,
            itemCount: indices.length,
            physics: const NeverScrollableScrollPhysics(),
            shrinkWrap: true,
            itemBuilder: (buildContext, i) {
              final idx = indices[i];
              List<Widget> children = [
                Flexible(
                  child: ListTile(
                    title: Text(
                      widget.input[idx],
                      textAlign: TextAlign.center,
                    ),
                    visualDensity: VisualDensity.compact,
                  ),
                )
              ];
              if (hasTr) {
                children.add(
                  Flexible(
                    flex: 1,
                    child: ListTile(
                      title: buildTrText(
                          widget.input[idx], trResults["text"][idx]),
                      visualDensity: VisualDensity.compact,
                    ),
                  ),
                );
              }
              if (hasSedw) {
                children.addAll([
                  Flexible(
                    flex: 1,
                    child: ListTile(
                      title: buildSedwText(sedwResults["text"][idx],
                          sedwResults["detections"][idx]),
                      visualDensity: VisualDensity.compact,
                    ),
                  ),
                ]);
              }
              if (hasSec) {
                children.addAll([
                  Flexible(
                    flex: 1,
                    child: ListTile(
                      title: buildSecText(secResults["text"][idx],
                          secResults["edited"]["text"][idx]),
                      visualDensity: VisualDensity.compact,
                    ),
                  ),
                ]);
              }
              return Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    child: Row(
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

  Widget buildTrText(String a, String b) {
    a = a.trim().replaceAll(RegExp(r"\s+"), " ");
    b = b.trim().replaceAll(RegExp(r"\s+"), " ");
    if (a.replaceAll(" ", "") != b.replaceAll(" ", "")) {
      // if a and b differ in something else than whitespaces return unformatted text
      // should not happen but just in case

      return Text(b);
    }
    List<TextSpan> children = [];
    int aPtr = 0;
    int bPtr = 0;
    while (aPtr < a.length || bPtr < b.length) {
      final aChar = a[aPtr];
      final bChar = b[bPtr];

      if (aChar == bChar) {
        children.add(TextSpan(text: bChar));
        aPtr++;
        bPtr++;
      } else if (aChar == " ") {
        if (children.isNotEmpty) {
          children.last = TextSpan(
              text: b[bPtr - 1], style: const TextStyle(color: uniRed, fontWeight: FontWeight.bold));
        }
        children
            .add(TextSpan(text: bChar, style: const TextStyle(color: uniRed, fontWeight: FontWeight.bold)));
        aPtr += 2;
        bPtr++;
      } else if (bChar == " ") {
        children.add(TextSpan(
            text: bChar, style: const TextStyle(backgroundColor: uniGreen)));
        bPtr++;
      } else {
        // should not happen
        throw "should not happen";
      }
    }
    return Text.rich(TextSpan(children: children), textAlign: TextAlign.center);
  }

  Widget buildSedwText(String text, List<dynamic> detections) {
    final words = text.trim().split(" ");
    List<InlineSpan> children = [];
    for (int i = 0; i < words.length; i++) {
      int isError = i < detections.length ? detections[i] : 0;
      final word = (i > 0 ? " " : "") + words[i].trim();
      if (isError == 1) {
        children.add(
          TextSpan(
            text: word,
            style: const TextStyle(color: uniRed, fontWeight: FontWeight.bold),
          ),
        );
      } else {
        children.add(TextSpan(text: word));
      }
    }
    return Text.rich(
      TextSpan(children: children),
      textAlign: TextAlign.center,
    );
  }

  Widget buildSecText(String text, List<dynamic> edited) {
    final words = text.trim().split(" ");
    List<InlineSpan> children = [];
    for (int i = 0; i < words.length; i++) {
      bool isEdited = edited.contains(i);
      final word = (i > 0 ? " " : "") + words[i].trim();
      if (isEdited) {
        children.add(
          TextSpan(
            text: word,
            style:
                const TextStyle(color: uniOrange, fontWeight: FontWeight.bold),
          ),
        );
      } else {
        children.add(TextSpan(text: word));
      }
    }
    return Text.rich(
      TextSpan(children: children),
      textAlign: TextAlign.center,
    );
  }

  Widget resultActions(BuildContext context, String taskName, String rawOutput,
      String downloadFileName) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        IconButton(
          onPressed: () async {
            await Clipboard.setData(ClipboardData(text: rawOutput));
            if (mounted) {
              showMessage(
                  context,
                  Message(
                      "Copied $taskName outputs to clipboard", Status.info));
            }
          },
          tooltip: "Copy raw $taskName outputs to clipboard",
          splashRadius: 16,
          icon: const Icon(Icons.content_copy),
        ),
        IconButton(
          onPressed: () async {
            final downloader = FileDownloader();
            final fileName =
                await downloader.downloadFile(rawOutput, downloadFileName);
            if (mounted) {
              if (fileName != null) {
                showMessage(
                  context,
                  Message(
                      "Downloaded $taskName outputs to $fileName", Status.info),
                );
              } else {
                showMessage(
                    context,
                    Message("Unexpected error downloading $taskName outputs",
                        Status.error));
              }
            }
          },
          tooltip: "Download raw $taskName outputs",
          splashRadius: 16,
          icon: const Icon(Icons.file_download),
        )
      ],
    );
  }
}
