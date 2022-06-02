import 'package:flutter/material.dart';
import 'package:webapp/base_view.dart';
import 'package:webapp/colors.dart';
import 'package:webapp/home_model.dart';

class HomeView extends StatefulWidget {
  const HomeView({super.key});

  @override
  State<HomeView> createState() => _HomeViewState();
}

Widget wrapScaffold(Widget widget) {
  return SafeArea(child: Scaffold(body: widget));
}

Widget wrapPadding(Widget widget) {
  return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 8),
      child: widget);
}

class _HomeViewState extends State<HomeView> {
  final GlobalKey<FormFieldState> _trModelKey = GlobalKey();
  final GlobalKey<FormFieldState> _sedwModelKey = GlobalKey();
  final GlobalKey<FormFieldState> _secModelKey = GlobalKey();
  final GlobalKey<FormFieldState> _textKey = GlobalKey();

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
                return wrapPadding(
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisAlignment: MainAxisAlignment.start,
                    children: [
                      const Text(
                        "Detect and correct spelling errors in text with space errors.",
                        style: TextStyle(fontSize: 22),
                      ),
                      const Divider(height: 32),
                      const Text(
                        "Select a model pipeline:",
                        style: TextStyle(fontSize: 16),
                      ),
                      const SizedBox(height: 16),
                      Row(
                        children: [
                          Flexible(
                            child: Row(
                              children: [
                                Flexible(
                                  child: DropdownButtonFormField<String>(
                                    key: _trModelKey,
                                    decoration: const InputDecoration(
                                        labelText: "Tokenization repair model"),
                                    icon: const Icon(
                                        Icons.arrow_drop_down_rounded),
                                    items: model
                                        .models("tokenization repair")
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
                                IconButton(
                                  splashRadius: 16,
                                  color: uniRed,
                                  icon: const Icon(Icons.clear),
                                  onPressed: () {
                                    setState(() {
                                      _trModelKey.currentState!.reset();
                                      model.trModel = "";
                                    });
                                  },
                                )
                              ],
                            ),
                          ),
                          const Icon(Icons.arrow_forward),
                          Flexible(
                            child: Row(
                              children: [
                                Flexible(
                                  child: DropdownButtonFormField<String>(
                                    key: _sedwModelKey,
                                    decoration: const InputDecoration(
                                        labelText:
                                            "Word-level spelling error detection model"),
                                    icon: const Icon(
                                        Icons.arrow_drop_down_rounded),
                                    items: model
                                        .models("sed words")
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
                                IconButton(
                                  splashRadius: 16,
                                  color: uniRed,
                                  icon: const Icon(Icons.clear),
                                  onPressed: () {
                                    setState(() {
                                      _sedwModelKey.currentState!.reset();
                                      model.sedwModel = "";
                                    });
                                  },
                                )
                              ],
                            ),
                          ),
                          const Icon(Icons.arrow_forward),
                          Flexible(
                            child: Row(
                              children: [
                                Flexible(
                                  child: DropdownButtonFormField<String>(
                                    key: _secModelKey,
                                    decoration: const InputDecoration(
                                        labelText:
                                            "Spelling error correction model"),
                                    icon: const Icon(
                                        Icons.arrow_drop_down_rounded),
                                    items: model
                                        .models("sec")
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
                                          model.secModel = modelName!;
                                          if (modelName ==
                                              "tokenization repair++") {
                                            model.trModel = "";
                                            model.sedwModel = "";
                                            model.addMessage(Message(
                                                "Tokenization repair++ model can perform tokenization repair and spelling error detection by itself",
                                                Status.info));
                                          }
                                        },
                                      );
                                    },
                                  ),
                                ),
                                IconButton(
                                  splashRadius: 16,
                                  color: uniRed,
                                  icon: const Icon(Icons.clear),
                                  onPressed: () {
                                    setState(() {
                                      _secModelKey.currentState!.reset();
                                      model.secModel = "";
                                    });
                                  },
                                )
                              ],
                            ),
                          )
                        ],
                      ),
                      const Divider(height: 32),
                      IntrinsicHeight(
                        child: Row(
                          children: [
                            Flexible(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  TextFormField(
                                    key: _textKey,
                                    maxLines: 10,
                                    decoration: const InputDecoration(
                                      border: OutlineInputBorder(),
                                      hintText:
                                          "Enter some text to spell check",
                                    ),
                                    enabled: model.validPipeline,
                                    onChanged: (String? input) async {
                                      if (model.live &&
                                          !model.waiting &&
                                          input != null) {
                                        model.runPipeline(input);
                                      }
                                    },
                                  ),
                                  const SizedBox(height: 8),
                                  Row(
                                    mainAxisAlignment: MainAxisAlignment.start,
                                    children: [
                                      ElevatedButton.icon(
                                        onPressed: model.validPipeline &&
                                                !model.waiting &&
                                                !model.live
                                            ? () {
                                                setState(() {
                                                  model.runPipeline(_textKey
                                                      .currentState?.value);
                                                });
                                              }
                                            : null,
                                        icon: const Icon(Icons.edit_note),
                                        label: const Text("Run pipeline"),
                                      ),
                                      const SizedBox(width: 8),
                                      ElevatedButton.icon(
                                        onPressed: model.validPipeline
                                            ? () {
                                                setState(() {
                                                  model.live = !model.live;
                                                });
                                              }
                                            : null,
                                        icon: const Icon(Icons.stream),
                                        label: Text(model.live
                                            ? "Disable live pipeline"
                                            : "Enable live pipeline"),
                                      ),
                                    ],
                                  )
                                ],
                              ),
                            ),
                            const VerticalDivider(width: 32),
                            const Flexible(child: Text("bla"))
                          ],
                        ),
                      ),
                      Text(model.resultText)
                    ].where((Widget? element) => element != null).toList(),
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
