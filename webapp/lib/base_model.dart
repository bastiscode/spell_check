import 'package:flutter/widgets.dart';

enum ViewState {
  Idle,
  Busy
}

class BaseModel extends ChangeNotifier {
  ViewState _state = ViewState.Idle;

  ViewState get state => _state;

  bool _disposed = false;

  bool get disposed => _disposed;

  @override
  void dispose() {
    _disposed = true;
    super.dispose();
  }

  @override
  void notifyListeners() {
    if (!_disposed) {
      super.notifyListeners();
    }
  }

  void setState(ViewState viewState) {
    _state = viewState;
    notifyListeners();
  }
}
