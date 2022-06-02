import 'package:get_it/get_it.dart';
import 'package:webapp/home_model.dart';

GetIt locator = GetIt.instance;

void setupLocator() {
  locator.registerFactory(() => HomeModel());
}