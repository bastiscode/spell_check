import 'package:collection/collection.dart';
import 'package:flutter/material.dart';
import 'package:webapp/colors.dart';

class Preset {
  final IconData icon;
  final String label;
  final String? trModel;
  final String? sedwModel;
  final String? secModel;

  const Preset(this.icon, this.label,
      {this.trModel, this.sedwModel, this.secModel});
}

typedef OnSelected = Function(Preset?);

class Presets extends StatefulWidget {
  final List<Preset> presets;
  final dynamic models;
  final OnSelected onSelected;

  const Presets(
      {required this.presets,
      required this.models,
      required this.onSelected,
      super.key});

  @override
  State<StatefulWidget> createState() => _PresetsState();
}

class _PresetsState extends State<Presets> {
  int selectedIdx = -1;

  @override
  Widget build(BuildContext context) {
    return Wrap(
        runSpacing: 8,
        spacing: 8,
        children:
            widget.presets.where((element) => true).mapIndexed((idx, preset) {
          final selected = selectedIdx == idx;
          return InputChip(
            avatar: Icon(preset.icon, color: selected ? Colors.white : Colors.black),
            label: Text(preset.label),
            labelStyle: TextStyle(color: selected ? Colors.white : Colors.black),
            visualDensity: VisualDensity.compact,
            selected: selected,
            selectedColor: uniBlue,
            onSelected: (isSelected) {
              widget.onSelected(isSelected ? preset : null);
              setState(() {
                if (isSelected) {
                  selectedIdx = idx;
                } else {
                  selectedIdx = -1;
                }
              });
            },
          );
        }).toList());
  }
}
