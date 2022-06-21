String formatS(double s) {
  if (s < 1) {
    final ms = s * 1000;
    return "${ms.round()}ms";
  } else {
    return "${s.toStringAsFixed(2)}s";
  }
}

String formatBps(double bps) {
  if (bps > 1000) {
    bps /= 1000;
    return "${bps.toStringAsFixed(2)}kB/s";
  } else {
    return "${bps.round()}b/s";
  }
}
