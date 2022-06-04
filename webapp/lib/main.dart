import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:webapp/colors.dart';
import 'package:webapp/home_view.dart';
import 'package:webapp/locator.dart';

void main() {
  setupLocator();
  runApp(const SpellCheckApp());
}

class SpellCheckApp extends StatelessWidget {
  const SpellCheckApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: "Spell checking",
      theme: ThemeData(
          brightness: Brightness.light,
          primaryColor: uniBlue,
          colorScheme: ColorScheme.fromSeed(seedColor: uniBlue),
          inputDecorationTheme:
              const InputDecorationTheme(border: OutlineInputBorder()),
          fontFamily: "Georgia"),
      home: const HomeView(),
    );
  }
}
