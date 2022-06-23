import 'package:collection/collection.dart';
import 'package:flutter/material.dart';
import 'package:webapp/colors.dart';

class Preset {
  final IconData? icon;
  final String label;
  final String? trModel;
  final String? sedwModel;
  final String? secModel;

  const Preset(this.label,
      {this.trModel, this.sedwModel, this.secModel, this.icon});
}

typedef OnSelected = Function(Preset?);

class Presets extends StatefulWidget {
  final List<Preset> presets;
  final dynamic models;
  final String? trModel;
  final String? sedwModel;
  final String? secModel;
  final OnSelected onSelected;

  const Presets(
      {required this.presets,
      required this.models,
      required this.trModel,
      required this.sedwModel,
      required this.secModel,
      required this.onSelected,
      super.key});

  @override
  State<StatefulWidget> createState() => _PresetsState();
}

class PresetInfo {
  final bool valid;
  final bool matching;

  PresetInfo(this.valid, this.matching);
}

class _PresetsState extends State<Presets> {
  int selectedIdx = -1;

  PresetInfo presetInfo(Preset preset) {
    bool valid = true;
    bool matching = preset.trModel == widget.trModel &&
        preset.sedwModel == widget.sedwModel &&
        preset.secModel == widget.secModel;
    if (preset.trModel != null &&
        widget.models.firstWhere(
                (element) =>
                    element["task"] == "tokenization repair" &&
                    element["name"] == preset.trModel!,
                orElse: () => null) ==
            null) {
      valid = false;
    }
    if (preset.sedwModel != null &&
        widget.models.firstWhere(
                (element) =>
                    element["task"] == "sed words" &&
                    element["name"] == preset.sedwModel!,
                orElse: () => null) ==
            null) {
      valid = false;
    }
    if (preset.secModel != null &&
        widget.models.firstWhere(
                (element) =>
                    element["task"] == "sec" &&
                    element["name"] == preset.secModel!,
                orElse: () => null) ==
            null) {
      valid = false;
    }
    return PresetInfo(valid, matching);
  }

  @override
  Widget build(BuildContext context) {
    final presetInfos = widget.presets.map((preset) => presetInfo(preset));
    return Wrap(
      runSpacing: 8,
      spacing: 8,
      children:
          presetInfos.where((info) => info.valid).mapIndexed(
        (idx, info) {
          final preset = widget.presets[idx];
          return ChoiceChip(
            label: Text(preset.label),
            labelStyle:
                TextStyle(color: info.matching ? Colors.white : Colors.black),
            visualDensity: VisualDensity.compact,
            selected: info.matching,
            selectedColor: uniBlue,
            onSelected: (_) {
              setState(() {
                widget.onSelected(preset);
              });
            },
          );
        },
      ).toList(),
    );
  }
}
