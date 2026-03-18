import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:shared_preferences/shared_preferences.dart';

// ─────────────────────────────────────────────────────────────────────────────
// Theme
// ─────────────────────────────────────────────────────────────────────────────
const Color kBg         = Color(0xFF080B12);
const Color kSurface    = Color(0xFF0E1420);
const Color kCard       = Color(0xFF131926);
const Color kNeon       = Color(0xFF00F5FF);
const Color kNeonGreen  = Color(0xFF00FF88);
const Color kNeonRed    = Color(0xFFFF2D55);
const Color kNeonPurple = Color(0xFFBF5AF2);
const Color kText       = Color(0xFFE8EDF5);
const Color kSubtext    = Color(0xFF6B7A99);

// ─────────────────────────────────────────────────────────────────────────────
// Settings Manager
// ─────────────────────────────────────────────────────────────────────────────
class AppSettings {
  static const String _keyUrl    = 'backend_url';
  // ── CHANGE THIS to your Railway URL after deployment ──────────────────────
  static const String defaultUrl = 'https://YOUR-APP.railway.app';

  static Future<String> getBackendUrl() async {
    final prefs = await SharedPreferences.getInstance();
    return prefs.getString(_keyUrl) ?? defaultUrl;
  }

  static Future<void> setBackendUrl(String url) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_keyUrl, url.trim().replaceAll(RegExp(r'/$'), ''));
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Scan History Model
// ─────────────────────────────────────────────────────────────────────────────
class ScanRecord {
  final String filename;
  final String label;
  final double confidence;
  final String type;
  final DateTime timestamp;

  ScanRecord({
    required this.filename,
    required this.label,
    required this.confidence,
    required this.type,
    required this.timestamp,
  });
}

final List<ScanRecord> scanHistory = [];

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
void main() {
  runApp(const SusAIApp());
}

class SusAIApp extends StatelessWidget {
  const SusAIApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SusAI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: kBg,
        fontFamily: 'monospace',
        colorScheme: const ColorScheme.dark(primary: kNeon, surface: kSurface),
      ),
      home: const SplashScreen(),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Logo Widget — swaps between asset logo and fallback icon
// ─────────────────────────────────────────────────────────────────────────────
class AppLogo extends StatelessWidget {
  final double size;
  final bool glow;
  final double glowOpacity;

  const AppLogo({super.key, this.size = 80, this.glow = false, this.glowOpacity = 0.6});

  @override
  Widget build(BuildContext context) {
    // ── LOGO: replace Icon with Image.asset once you add your logo file ──────
    // Step 1: Add logo.png to frontend/assets/logo.png
    // Step 2: Uncomment the Image.asset line below and comment out the Icon line
    // ─────────────────────────────────────────────────────────────────────────
    final Widget logo = Icon(
      Icons.remove_red_eye_outlined,
      size: size,
      color: kNeon,
    );
    // final Widget logo = Image.asset('assets/logo.png', width: size, height: size);

    if (!glow) return logo;

    return Container(
      width: size * 1.5,
      height: size * 1.5,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        boxShadow: [BoxShadow(color: kNeon.withOpacity(glowOpacity), blurRadius: 40, spreadRadius: 10)],
      ),
      child: Center(child: logo),
    );
  }
}

// Small logo for app bar / home header
class AppLogoSmall extends StatelessWidget {
  final double size;
  const AppLogoSmall({super.key, this.size = 28});

  @override
  Widget build(BuildContext context) {
    // ── LOGO: same swap as above ─────────────────────────────────────────────
    return Icon(Icons.remove_red_eye_outlined, color: kNeon, size: size);
    // return Image.asset('assets/logo.png', width: size, height: size);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Splash Screen
// ─────────────────────────────────────────────────────────────────────────────
class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});
  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> with TickerProviderStateMixin {
  late AnimationController _glowCtrl;
  late AnimationController _fadeCtrl;
  late Animation<double> _glowAnim;
  late Animation<double> _fadeAnim;

  @override
  void initState() {
    super.initState();
    _glowCtrl = AnimationController(duration: const Duration(milliseconds: 1500), vsync: this)..repeat(reverse: true);
    _fadeCtrl = AnimationController(duration: const Duration(milliseconds: 800), vsync: this)..forward();
    _glowAnim = Tween<double>(begin: 0.4, end: 1.0).animate(CurvedAnimation(parent: _glowCtrl, curve: Curves.easeInOut));
    _fadeAnim = Tween<double>(begin: 0.0, end: 1.0).animate(CurvedAnimation(parent: _fadeCtrl, curve: Curves.easeOut));

    Future.delayed(const Duration(seconds: 3), () {
      if (mounted) {
        Navigator.pushReplacement(context,
          PageRouteBuilder(
            pageBuilder: (_, __, ___) => const HomeScreen(),
            transitionsBuilder: (_, anim, __, child) => FadeTransition(opacity: anim, child: child),
            transitionDuration: const Duration(milliseconds: 600),
          ),
        );
      }
    });
  }

  @override
  void dispose() { _glowCtrl.dispose(); _fadeCtrl.dispose(); super.dispose(); }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: kBg,
      body: Center(
        child: FadeTransition(
          opacity: _fadeAnim,
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // ── LOGO with glow animation ───────────────────────────────────
              AnimatedBuilder(
                animation: _glowAnim,
                builder: (_, __) => AppLogo(size: 80, glow: true, glowOpacity: _glowAnim.value * 0.6),
              ),
              const SizedBox(height: 32),
              ShaderMask(
                shaderCallback: (b) => const LinearGradient(colors: [kNeon, kNeonPurple]).createShader(b),
                child: const Text('SusAI', style: TextStyle(fontSize: 52, fontWeight: FontWeight.w900, letterSpacing: 6, color: Colors.white)),
              ),
              const Text('AI MEDIA DETECTOR', style: TextStyle(fontSize: 12, letterSpacing: 6, color: kSubtext)),
              const SizedBox(height: 48),
              SizedBox(width: 120, child: LinearProgressIndicator(backgroundColor: kSurface, valueColor: const AlwaysStoppedAnimation<Color>(kNeon))),
              const SizedBox(height: 16),
              const Text('INITIALIZING AI ENGINE...', style: TextStyle(fontSize: 10, letterSpacing: 3, color: kSubtext)),
            ],
          ),
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Home Screen
// ─────────────────────────────────────────────────────────────────────────────
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  String _backendUrl = AppSettings.defaultUrl;
  bool _backendOnline = false;

  @override
  void initState() { super.initState(); _loadAndCheck(); }

  Future<void> _loadAndCheck() async {
    final url = await AppSettings.getBackendUrl();
    setState(() => _backendUrl = url);
    _checkBackend(url);
  }

  Future<void> _checkBackend(String url) async {
    try {
      final r = await http.get(Uri.parse('$url/health')).timeout(const Duration(seconds: 5));
      setState(() => _backendOnline = r.statusCode == 200);
    } catch (_) { setState(() => _backendOnline = false); }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: kBg,
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 16),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Row(children: [
                    // ── LOGO in header ───────────────────────────────────────
                    const AppLogoSmall(size: 28),
                    const SizedBox(width: 12),
                    ShaderMask(
                      shaderCallback: (b) => const LinearGradient(colors: [kNeon, kNeonPurple]).createShader(b),
                      child: const Text('SusAI', style: TextStyle(fontSize: 26, fontWeight: FontWeight.w900, letterSpacing: 3, color: Colors.white)),
                    ),
                  ]),
                  IconButton(
                    icon: const Icon(Icons.settings_outlined, color: kSubtext),
                    onPressed: () async {
                      await Navigator.push(context, MaterialPageRoute(builder: (_) => const SettingsScreen()));
                      _loadAndCheck();
                    },
                  ),
                ],
              ),
              const Text('AI-powered media authenticity detection', style: TextStyle(color: kSubtext, fontSize: 12, letterSpacing: 1)),
              const SizedBox(height: 40),

              _menuCard(context, icon: Icons.image_search, title: 'IMAGE SCAN', subtitle: 'Detect AI-generated images', color: kNeon,
                  onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const ImageScanScreen()))),
              const SizedBox(height: 16),
              _menuCard(context, icon: Icons.videocam_outlined, title: 'VIDEO SCAN', subtitle: 'Analyze video frame by frame', color: kNeonPurple,
                  onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const VideoScanScreen()))),
              const SizedBox(height: 16),
              _menuCard(context, icon: Icons.history, title: 'SCAN HISTORY', subtitle: 'View previous detections', color: kNeonGreen,
                  onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const HistoryScreen()))),

              const Spacer(),
              GestureDetector(
                onTap: () => _checkBackend(_backendUrl),
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                  decoration: BoxDecoration(
                    color: kCard, borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: (_backendOnline ? kNeonGreen : kNeonRed).withOpacity(0.3)),
                  ),
                  child: Row(children: [
                    Container(width: 8, height: 8,
                      decoration: BoxDecoration(
                        color: _backendOnline ? kNeonGreen : kNeonRed, shape: BoxShape.circle,
                        boxShadow: [BoxShadow(color: (_backendOnline ? kNeonGreen : kNeonRed).withOpacity(0.6), blurRadius: 6)],
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(child: Text(
                      _backendOnline ? 'AI ENGINE ONLINE  •  EfficientNetB0' : 'BACKEND OFFLINE  •  TAP TO RETRY',
                      style: TextStyle(color: _backendOnline ? kNeonGreen : kNeonRed, fontSize: 11, letterSpacing: 1),
                    )),
                    Icon(Icons.refresh, color: (_backendOnline ? kNeonGreen : kNeonRed).withOpacity(0.6), size: 16),
                  ]),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _menuCard(BuildContext context, {required IconData icon, required String title, required String subtitle, required Color color, required VoidCallback onTap}) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(color: kCard, borderRadius: BorderRadius.circular(16), border: Border.all(color: color.withOpacity(0.3)), boxShadow: [BoxShadow(color: color.withOpacity(0.05), blurRadius: 20)]),
        child: Row(children: [
          Container(width: 52, height: 52,
            decoration: BoxDecoration(color: color.withOpacity(0.1), borderRadius: BorderRadius.circular(12), border: Border.all(color: color.withOpacity(0.4))),
            child: Icon(icon, color: color, size: 26),
          ),
          const SizedBox(width: 16),
          Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(title, style: TextStyle(color: color, fontSize: 14, fontWeight: FontWeight.bold, letterSpacing: 2)),
            const SizedBox(height: 4),
            Text(subtitle, style: const TextStyle(color: kSubtext, fontSize: 12)),
          ])),
          Icon(Icons.arrow_forward_ios, color: color.withOpacity(0.5), size: 16),
        ]),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Settings Screen
// ─────────────────────────────────────────────────────────────────────────────
class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});
  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late TextEditingController _ctrl;
  bool _isTesting = false;
  String? _testResult;
  bool? _testSuccess;

  @override
  void initState() {
    super.initState();
    _ctrl = TextEditingController();
    AppSettings.getBackendUrl().then((url) => _ctrl.text = url);
  }

  @override
  void dispose() { _ctrl.dispose(); super.dispose(); }

  Future<void> _test() async {
    setState(() { _isTesting = true; _testResult = null; });
    try {
      final r = await http.get(Uri.parse('${_ctrl.text}/health')).timeout(const Duration(seconds: 5));
      setState(() { _testSuccess = r.statusCode == 200; _testResult = r.statusCode == 200 ? '✓ Connected successfully!' : '✗ Server error ${r.statusCode}'; });
    } catch (_) {
      setState(() { _testSuccess = false; _testResult = '✗ Cannot reach server. Check the URL and make sure backend is running.'; });
    } finally { setState(() => _isTesting = false); }
  }

  Future<void> _save() async {
    await AppSettings.setBackendUrl(_ctrl.text);
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Settings saved!'), backgroundColor: kNeonGreen));
      Navigator.pop(context);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: kBg,
      appBar: _buildAppBar('SETTINGS', kNeonPurple),
      body: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          const Text('BACKEND URL', style: TextStyle(color: kSubtext, fontSize: 11, letterSpacing: 2)),
          const SizedBox(height: 12),
          TextField(
            controller: _ctrl,
            style: const TextStyle(color: kText, fontSize: 14),
            keyboardType: TextInputType.url,
            decoration: InputDecoration(
              hintText: 'https://your-app.railway.app',
              hintStyle: const TextStyle(color: kSubtext),
              filled: true, fillColor: kCard,
              prefixIcon: const Icon(Icons.link, color: kNeonPurple, size: 20),
              border: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide(color: kNeonPurple.withOpacity(0.3))),
              enabledBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: BorderSide(color: kNeonPurple.withOpacity(0.3))),
              focusedBorder: OutlineInputBorder(borderRadius: BorderRadius.circular(12), borderSide: const BorderSide(color: kNeonPurple)),
            ),
          ),
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(color: kNeon.withOpacity(0.05), borderRadius: BorderRadius.circular(10), border: Border.all(color: kNeon.withOpacity(0.2))),
            child: const Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text('BACKEND URL OPTIONS:', style: TextStyle(color: kNeon, fontSize: 10, letterSpacing: 1)),
              SizedBox(height: 8),
              Text(
                '🌐 Railway (recommended):\n'
                '   https://your-app.railway.app\n\n'
                '🏠 Local (same WiFi only):\n'
                '   http://192.168.1.YOUR_IP:8000\n'
                '   Run: uvicorn app.main:app --host 0.0.0.0',
                style: TextStyle(color: kSubtext, fontSize: 11, height: 1.8),
              ),
            ]),
          ),
          const SizedBox(height: 20),
          SizedBox(
            width: double.infinity, height: 48,
            child: OutlinedButton(
              onPressed: _isTesting ? null : _test,
              style: OutlinedButton.styleFrom(side: const BorderSide(color: kNeon), shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12))),
              child: _isTesting
                  ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2, color: kNeon))
                  : const Text('TEST CONNECTION', style: TextStyle(color: kNeon, letterSpacing: 2, fontSize: 13)),
            ),
          ),
          if (_testResult != null) ...[
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(color: (_testSuccess! ? kNeonGreen : kNeonRed).withOpacity(0.08), borderRadius: BorderRadius.circular(10), border: Border.all(color: (_testSuccess! ? kNeonGreen : kNeonRed).withOpacity(0.4))),
              child: Text(_testResult!, style: TextStyle(color: _testSuccess! ? kNeonGreen : kNeonRed, fontSize: 12)),
            ),
          ],
          const SizedBox(height: 20),
          SizedBox(
            width: double.infinity, height: 54,
            child: ElevatedButton(
              onPressed: _save,
              style: ElevatedButton.styleFrom(backgroundColor: kNeonPurple, foregroundColor: Colors.white, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)), elevation: 0),
              child: const Text('SAVE SETTINGS', style: TextStyle(fontWeight: FontWeight.bold, letterSpacing: 3)),
            ),
          ),
        ]),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Image Scan Screen
// ─────────────────────────────────────────────────────────────────────────────
class ImageScanScreen extends StatefulWidget {
  const ImageScanScreen({super.key});
  @override
  State<ImageScanScreen> createState() => _ImageScanScreenState();
}

class _ImageScanScreenState extends State<ImageScanScreen> {
  File? _img;
  bool _loading = false;
  Map<String, dynamic>? _result;
  String? _error;
  final _picker = ImagePicker();

  Future<void> _pick(ImageSource src) async {
    final f = await _picker.pickImage(source: src);
    if (f != null) setState(() { _img = File(f.path); _result = null; _error = null; });
  }

  Future<void> _analyze() async {
    if (_img == null) return;
    setState(() { _loading = true; _error = null; });
    try {
      final url = await AppSettings.getBackendUrl();
      final req = http.MultipartRequest('POST', Uri.parse('$url/predict/image'));
      req.files.add(await http.MultipartFile.fromPath('file', _img!.path));
      final res = await req.send().timeout(const Duration(seconds: 30));
      final data = jsonDecode(await res.stream.bytesToString());
      if (res.statusCode == 200) {
        setState(() => _result = data);
        scanHistory.insert(0, ScanRecord(filename: data['filename'] ?? 'image', label: data['label'], confidence: (data['confidence'] as num).toDouble(), type: 'image', timestamp: DateTime.now()));
      } else { setState(() => _error = 'Server error: ${res.statusCode}'); }
    } catch (_) { setState(() => _error = 'Connection failed. Go to Settings ⚙ and check the backend URL.'); }
    finally { setState(() => _loading = false); }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: kBg,
      appBar: _buildAppBar('IMAGE SCAN', kNeon),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(children: [
          GestureDetector(
            onTap: () => _showSheet(),
            child: Container(
              height: 260, width: double.infinity,
              decoration: BoxDecoration(color: kCard, borderRadius: BorderRadius.circular(16), border: Border.all(color: _img != null ? kNeon.withOpacity(0.5) : kSubtext.withOpacity(0.2))),
              child: _img != null
                  ? ClipRRect(borderRadius: BorderRadius.circular(15), child: Image.file(_img!, fit: BoxFit.cover))
                  : Column(mainAxisAlignment: MainAxisAlignment.center, children: [
                      Icon(Icons.add_photo_alternate_outlined, size: 56, color: kNeon.withOpacity(0.5)),
                      const SizedBox(height: 12),
                      const Text('TAP TO SELECT IMAGE', style: TextStyle(color: kSubtext, letterSpacing: 2, fontSize: 12)),
                    ]),
            ),
          ),
          const SizedBox(height: 20),
          Row(children: [
            Expanded(child: _outlineBtn(icon: Icons.photo_library, label: 'GALLERY', color: kNeon, onTap: () => _pick(ImageSource.gallery))),
            const SizedBox(width: 12),
            Expanded(child: _outlineBtn(icon: Icons.camera_alt, label: 'CAMERA', color: kNeonPurple, onTap: () => _pick(ImageSource.camera))),
          ]),
          const SizedBox(height: 20),
          SizedBox(
            width: double.infinity, height: 54,
            child: ElevatedButton(
              onPressed: _img != null && !_loading ? _analyze : null,
              style: ElevatedButton.styleFrom(backgroundColor: kNeon, foregroundColor: kBg, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)), elevation: 0),
              child: _loading
                  ? const SizedBox(width: 22, height: 22, child: CircularProgressIndicator(strokeWidth: 2, color: kBg))
                  : const Text('ANALYZE IMAGE', style: TextStyle(fontWeight: FontWeight.bold, letterSpacing: 3)),
            ),
          ),
          const SizedBox(height: 24),
          if (_error != null) _errorCard(_error!),
          if (_result != null) _resultCard(_result!),
        ]),
      ),
    );
  }

  void _showSheet() {
    showModalBottomSheet(context: context, backgroundColor: kCard,
      shape: const RoundedRectangleBorder(borderRadius: BorderRadius.vertical(top: Radius.circular(20))),
      builder: (_) => Padding(padding: const EdgeInsets.all(24), child: Column(mainAxisSize: MainAxisSize.min, children: [
        ListTile(leading: const Icon(Icons.photo_library, color: kNeon), title: const Text('Gallery', style: TextStyle(color: kText)),
            onTap: () { Navigator.pop(context); _pick(ImageSource.gallery); }),
        ListTile(leading: const Icon(Icons.camera_alt, color: kNeonPurple), title: const Text('Camera', style: TextStyle(color: kText)),
            onTap: () { Navigator.pop(context); _pick(ImageSource.camera); }),
      ])),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Video Scan Screen
// ─────────────────────────────────────────────────────────────────────────────
class VideoScanScreen extends StatefulWidget {
  const VideoScanScreen({super.key});
  @override
  State<VideoScanScreen> createState() => _VideoScanScreenState();
}

class _VideoScanScreenState extends State<VideoScanScreen> {
  File? _video;
  bool _loading = false;
  Map<String, dynamic>? _result;
  String? _error;
  final _picker = ImagePicker();

  Future<void> _pick() async {
    final f = await _picker.pickVideo(source: ImageSource.gallery);
    if (f != null) setState(() { _video = File(f.path); _result = null; _error = null; });
  }

  Future<void> _analyze() async {
    if (_video == null) return;
    setState(() { _loading = true; _error = null; });
    try {
      final url = await AppSettings.getBackendUrl();
      final req = http.MultipartRequest('POST', Uri.parse('$url/predict/video'));
      req.files.add(await http.MultipartFile.fromPath('file', _video!.path));
      final res = await req.send().timeout(const Duration(seconds: 120));
      final data = jsonDecode(await res.stream.bytesToString());
      if (res.statusCode == 200) {
        setState(() => _result = data);
        scanHistory.insert(0, ScanRecord(filename: data['filename'] ?? 'video', label: data['verdict'], confidence: (data['avg_fake_confidence'] as num).toDouble(), type: 'video', timestamp: DateTime.now()));
      } else { setState(() => _error = 'Server error: ${res.statusCode}'); }
    } catch (_) { setState(() => _error = 'Connection failed. Go to Settings ⚙ and check the backend URL.'); }
    finally { setState(() => _loading = false); }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: kBg,
      appBar: _buildAppBar('VIDEO SCAN', kNeonPurple),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(children: [
          GestureDetector(
            onTap: _pick,
            child: Container(
              height: 200, width: double.infinity,
              decoration: BoxDecoration(color: kCard, borderRadius: BorderRadius.circular(16), border: Border.all(color: _video != null ? kNeonPurple.withOpacity(0.5) : kSubtext.withOpacity(0.2))),
              child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
                Icon(_video != null ? Icons.video_file : Icons.video_library_outlined, size: 56, color: kNeonPurple.withOpacity(_video != null ? 1.0 : 0.5)),
                const SizedBox(height: 12),
                Text(_video != null ? _video!.path.split('/').last : 'TAP TO SELECT VIDEO',
                    style: TextStyle(color: _video != null ? kText : kSubtext, letterSpacing: 1, fontSize: 12), textAlign: TextAlign.center, maxLines: 2, overflow: TextOverflow.ellipsis),
                if (_video != null) ...[
                  const SizedBox(height: 8),
                  Text('${(_video!.lengthSync() / 1024 / 1024).toStringAsFixed(1)} MB', style: const TextStyle(color: kSubtext, fontSize: 11)),
                ],
              ]),
            ),
          ),
          const SizedBox(height: 20),
          _outlineBtn(icon: Icons.video_library, label: 'SELECT FROM GALLERY', color: kNeonPurple, onTap: _pick),
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(color: kNeonPurple.withOpacity(0.05), borderRadius: BorderRadius.circular(12), border: Border.all(color: kNeonPurple.withOpacity(0.2))),
            child: Row(children: [
              Icon(Icons.info_outline, color: kNeonPurple.withOpacity(0.7), size: 18),
              const SizedBox(width: 12),
              const Expanded(child: Text('Analyzes up to 20 frames per video. Larger files may take longer.', style: TextStyle(color: kSubtext, fontSize: 11, height: 1.5))),
            ]),
          ),
          const SizedBox(height: 20),
          SizedBox(
            width: double.infinity, height: 54,
            child: ElevatedButton(
              onPressed: _video != null && !_loading ? _analyze : null,
              style: ElevatedButton.styleFrom(backgroundColor: kNeonPurple, foregroundColor: Colors.white, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)), elevation: 0),
              child: _loading
                  ? const Column(mainAxisAlignment: MainAxisAlignment.center, children: [
                      SizedBox(width: 22, height: 22, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white)),
                      SizedBox(height: 4),
                      Text('ANALYZING FRAMES...', style: TextStyle(fontSize: 10, letterSpacing: 2)),
                    ])
                  : const Text('ANALYZE VIDEO', style: TextStyle(fontWeight: FontWeight.bold, letterSpacing: 3)),
            ),
          ),
          const SizedBox(height: 24),
          if (_error != null) _errorCard(_error!),
          if (_result != null) _videoResultCard(_result!),
        ]),
      ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// History Screen
// ─────────────────────────────────────────────────────────────────────────────
class HistoryScreen extends StatelessWidget {
  const HistoryScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: kBg,
      appBar: _buildAppBar('SCAN HISTORY', kNeonGreen),
      body: scanHistory.isEmpty
          ? Center(child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
              Icon(Icons.history, size: 64, color: kSubtext.withOpacity(0.3)),
              const SizedBox(height: 16),
              const Text('NO SCANS YET', style: TextStyle(color: kSubtext, letterSpacing: 3, fontSize: 13)),
              const SizedBox(height: 8),
              const Text('Run an image or video scan to see history', style: TextStyle(color: kSubtext, fontSize: 11)),
            ]))
          : ListView.separated(
              padding: const EdgeInsets.all(24),
              itemCount: scanHistory.length,
              separatorBuilder: (_, __) => const SizedBox(height: 12),
              itemBuilder: (_, i) {
                final r = scanHistory[i];
                final isFake = r.label == 'FAKE';
                final color = isFake ? kNeonRed : kNeonGreen;
                return Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(color: kCard, borderRadius: BorderRadius.circular(12), border: Border.all(color: color.withOpacity(0.2))),
                  child: Row(children: [
                    Container(width: 44, height: 44,
                      decoration: BoxDecoration(color: color.withOpacity(0.1), borderRadius: BorderRadius.circular(10)),
                      child: Icon(r.type == 'video' ? Icons.video_file : Icons.image, color: color, size: 22),
                    ),
                    const SizedBox(width: 14),
                    Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                      Text(r.filename, style: const TextStyle(color: kText, fontSize: 13, fontWeight: FontWeight.w600), maxLines: 1, overflow: TextOverflow.ellipsis),
                      const SizedBox(height: 4),
                      Text('${r.timestamp.hour}:${r.timestamp.minute.toString().padLeft(2, '0')}  •  ${r.type.toUpperCase()}',
                          style: const TextStyle(color: kSubtext, fontSize: 10, letterSpacing: 1)),
                    ])),
                    Column(crossAxisAlignment: CrossAxisAlignment.end, children: [
                      Container(padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                        decoration: BoxDecoration(color: color.withOpacity(0.15), borderRadius: BorderRadius.circular(6)),
                        child: Text(r.label, style: TextStyle(color: color, fontSize: 11, fontWeight: FontWeight.bold, letterSpacing: 1)),
                      ),
                      const SizedBox(height: 4),
                      Text('${r.confidence.toStringAsFixed(1)}%', style: TextStyle(color: color.withOpacity(0.7), fontSize: 11)),
                    ]),
                  ]),
                );
              },
            ),
    );
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shared Widgets
// ─────────────────────────────────────────────────────────────────────────────

PreferredSizeWidget _buildAppBar(String title, Color color) => AppBar(
  backgroundColor: kSurface, elevation: 0,
  iconTheme: const IconThemeData(color: kText),
  title: Text(title, style: TextStyle(color: color, fontSize: 14, fontWeight: FontWeight.bold, letterSpacing: 3)),
  bottom: PreferredSize(preferredSize: const Size.fromHeight(1), child: Container(height: 1, color: color.withOpacity(0.2))),
);

Widget _outlineBtn({required IconData icon, required String label, required Color color, required VoidCallback onTap}) =>
  GestureDetector(
    onTap: onTap,
    child: Container(
      height: 48,
      decoration: BoxDecoration(color: color.withOpacity(0.05), borderRadius: BorderRadius.circular(12), border: Border.all(color: color.withOpacity(0.4))),
      child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
        Icon(icon, color: color, size: 18), const SizedBox(width: 8),
        Text(label, style: TextStyle(color: color, fontSize: 12, fontWeight: FontWeight.bold, letterSpacing: 2)),
      ]),
    ),
  );

Widget _errorCard(String error) => Container(
  width: double.infinity, padding: const EdgeInsets.all(16),
  decoration: BoxDecoration(color: kNeonRed.withOpacity(0.08), borderRadius: BorderRadius.circular(12), border: Border.all(color: kNeonRed.withOpacity(0.4))),
  child: Row(children: [
    const Icon(Icons.error_outline, color: kNeonRed, size: 20), const SizedBox(width: 12),
    Expanded(child: Text(error, style: const TextStyle(color: kNeonRed, fontSize: 12))),
  ]),
);

Widget _resultCard(Map<String, dynamic> result) {
  final isFake = result['label'] == 'FAKE';
  final color = isFake ? kNeonRed : kNeonGreen;
  final conf = (result['confidence'] as num).toDouble();
  return Container(
    width: double.infinity, padding: const EdgeInsets.all(24),
    decoration: BoxDecoration(color: color.withOpacity(0.05), borderRadius: BorderRadius.circular(16), border: Border.all(color: color.withOpacity(0.4)), boxShadow: [BoxShadow(color: color.withOpacity(0.1), blurRadius: 20)]),
    child: Column(children: [
      Icon(isFake ? Icons.warning_amber_rounded : Icons.verified_rounded, size: 52, color: color),
      const SizedBox(height: 12),
      Text(isFake ? 'AI GENERATED' : 'AUTHENTIC', style: TextStyle(color: color, fontSize: 22, fontWeight: FontWeight.w900, letterSpacing: 4)),
      const SizedBox(height: 20),
      Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
        const Text('CONFIDENCE', style: TextStyle(color: kSubtext, fontSize: 10, letterSpacing: 2)),
        Text('${conf.toStringAsFixed(1)}%', style: TextStyle(color: color, fontSize: 13, fontWeight: FontWeight.bold)),
      ]),
      const SizedBox(height: 8),
      ClipRRect(borderRadius: BorderRadius.circular(4), child: LinearProgressIndicator(value: conf / 100, backgroundColor: kSurface, valueColor: AlwaysStoppedAnimation<Color>(color), minHeight: 8)),
      const SizedBox(height: 16),
      Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
        _statChip('RAW SCORE', result['raw_score'].toString()),
        _statChip('FILE', result['filename'] ?? '-'),
      ]),
    ]),
  );
}

Widget _videoResultCard(Map<String, dynamic> result) {
  final isFake = result['verdict'] == 'FAKE';
  final color = isFake ? kNeonRed : kNeonGreen;
  final fakeConf = (result['avg_fake_confidence'] as num).toDouble();
  final realConf = (result['avg_real_confidence'] as num).toDouble();
  return Container(
    width: double.infinity, padding: const EdgeInsets.all(24),
    decoration: BoxDecoration(color: color.withOpacity(0.05), borderRadius: BorderRadius.circular(16), border: Border.all(color: color.withOpacity(0.4)), boxShadow: [BoxShadow(color: color.withOpacity(0.1), blurRadius: 20)]),
    child: Column(children: [
      Icon(isFake ? Icons.warning_amber_rounded : Icons.verified_rounded, size: 52, color: color),
      const SizedBox(height: 12),
      Text(isFake ? 'DEEPFAKE DETECTED' : 'VIDEO AUTHENTIC', style: TextStyle(color: color, fontSize: 20, fontWeight: FontWeight.w900, letterSpacing: 3), textAlign: TextAlign.center),
      const SizedBox(height: 24),
      Row(children: [
        Expanded(child: _voteCard('FAKE FRAMES', result['fake_votes'].toString(), kNeonRed)),
        const SizedBox(width: 12),
        Expanded(child: _voteCard('REAL FRAMES', result['real_votes'].toString(), kNeonGreen)),
      ]),
      const SizedBox(height: 12),
      Row(children: [
        Expanded(child: _voteCard('FRAMES ANALYZED', result['frames_analyzed'].toString(), kNeon)),
        const SizedBox(width: 12),
        Expanded(child: _voteCard('PROCESS TIME', '${result['processing_seconds']}s', kNeonPurple)),
      ]),
      const SizedBox(height: 16),
      _confRow('FAKE CONFIDENCE', fakeConf, kNeonRed),
      const SizedBox(height: 8),
      _confRow('REAL CONFIDENCE', realConf, kNeonGreen),
    ]),
  );
}

Widget _voteCard(String label, String value, Color color) => Container(
  padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 12),
  decoration: BoxDecoration(color: color.withOpacity(0.08), borderRadius: BorderRadius.circular(10), border: Border.all(color: color.withOpacity(0.2))),
  child: Column(children: [
    Text(value, style: TextStyle(color: color, fontSize: 22, fontWeight: FontWeight.bold)),
    const SizedBox(height: 4),
    Text(label, style: const TextStyle(color: kSubtext, fontSize: 9, letterSpacing: 1), textAlign: TextAlign.center),
  ]),
);

Widget _confRow(String label, double value, Color color) => Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
  Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
    Text(label, style: const TextStyle(color: kSubtext, fontSize: 10, letterSpacing: 1)),
    Text('${value.toStringAsFixed(1)}%', style: TextStyle(color: color, fontSize: 11)),
  ]),
  const SizedBox(height: 4),
  ClipRRect(borderRadius: BorderRadius.circular(4), child: LinearProgressIndicator(value: value / 100, backgroundColor: kSurface, valueColor: AlwaysStoppedAnimation<Color>(color), minHeight: 6)),
]);

Widget _statChip(String label, String value) => Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
  Text(label, style: const TextStyle(color: kSubtext, fontSize: 9, letterSpacing: 1)),
  const SizedBox(height: 2),
  Text(value, style: const TextStyle(color: kSubtext, fontSize: 12, fontWeight: FontWeight.w600), maxLines: 1, overflow: TextOverflow.ellipsis),
]);