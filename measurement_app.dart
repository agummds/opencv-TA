import 'package:flutter/material.dart';
import 'package:mqtt_client/mqtt_client.dart';
import 'package:mqtt_client/mqtt_server_client.dart';
import 'dart:convert';

void main() {
  runApp(const MeasurementApp());
}

class MeasurementApp extends StatelessWidget {
  const MeasurementApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Pengukuran Real-time',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: const MeasurementPage(),
    );
  }
}

class MeasurementPage extends StatefulWidget {
  const MeasurementPage({super.key});

  @override
  State<MeasurementPage> createState() => _MeasurementPageState();
}

class _MeasurementPageState extends State<MeasurementPage> {
  final MqttServerClient client = MqttServerClient('broker.hivemq.com', 'flutter_client');
  String status = 'Menghubungkan...';
  double height = 0;
  double width = 0;
  String lastUpdate = 'Belum ada data';

  @override
  void initState() {
    super.initState();
    connectMQTT();
  }

  void connectMQTT() async {
    try {
      client.port = 1883;
      client.keepAlivePeriod = 60;
      client.secure = false;
      client.logging(on: true);

      // Koneksi ke broker
      await client.connect();
      
      if (client.connectionStatus?.state == MqttConnectionState.connected) {
        setState(() {
          status = 'Terhubung';
        });
        
        // Subscribe ke topic
        client.subscribe('raspberry/measurement', MqttQos.atLeastOnce);
        
        // Listen untuk pesan
        client.updates?.listen((List<MqttReceivedMessage<MqttMessage>> messages) {
          final message = messages[0].payload as MqttPublishMessage;
          final payload = MqttPublishPayload.bytesToStringAsString(message.payload.message);
          
          // Parse JSON data
          final data = jsonDecode(payload);
          setState(() {
            height = data['measurement']['height'];
            width = data['measurement']['width'];
            lastUpdate = DateTime.now().toString();
          });
        });
      } else {
        setState(() {
          status = 'Gagal terhubung';
        });
      }
    } catch (e) {
      setState(() {
        status = 'Error: $e';
      });
    }
  }

  @override
  void dispose() {
    client.disconnect();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Pengukuran Real-time'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              'Status: $status',
              style: const TextStyle(fontSize: 18),
            ),
            const SizedBox(height: 20),
            Card(
              margin: const EdgeInsets.all(16),
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  children: [
                    const Text(
                      'Hasil Pengukuran',
                      style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 20),
                    Text(
                      'Tinggi: ${height.toStringAsFixed(1)} cm',
                      style: const TextStyle(fontSize: 20),
                    ),
                    const SizedBox(height: 10),
                    Text(
                      'Lebar: ${width.toStringAsFixed(1)} cm',
                      style: const TextStyle(fontSize: 20),
                    ),
                    const SizedBox(height: 20),
                    Text(
                      'Update terakhir: $lastUpdate',
                      style: const TextStyle(fontSize: 12, color: Colors.grey),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
} 