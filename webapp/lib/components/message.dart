import 'package:flutter/material.dart';
import 'package:webapp/colors.dart';

enum Status { error, warn, info }

class Message {
  String message;
  Status status;

  Message(this.message, this.status);
}

void showMessage(BuildContext context, Message message) {
  ScaffoldMessenger.of(context).showSnackBar(
    SnackBar(
      content: Row(
        children: [
          Icon(
            message.status == Status.info
                ? Icons.info_outline
                : message.status == Status.warn
                ? Icons.warning_amber
                : Icons.error_outline,
            color: Colors.white,
          ),
          const SizedBox(width: 8),
          Flexible(
            child: Text(
              message.message,
              style: const TextStyle(color: Colors.white),
            ),
          ),
        ],
      ),
      backgroundColor: message.status == Status.info
          ? uniDarkGray
          : message.status == Status.warn
          ? uniOrange
          : uniRed,
      behavior: SnackBarBehavior.floating,
      margin: const EdgeInsets.all(8),
      padding: const EdgeInsets.all(8),
      duration: const Duration(seconds: 3),
    ),
  );
}
