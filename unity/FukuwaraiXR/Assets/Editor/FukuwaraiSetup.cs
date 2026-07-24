// 3D福笑い セットアップ自動化（UnityGaussianSplatting）
// メニュー: Fukuwarai/... から実行。
//  1. DX12 に設定（GSはDX12/Vulkan必須。設定後エディタ再起動）
//  2. Source の .ply から GaussianSplatAsset を生成
//  3. 各パーツを配置したシーンを構築（Renderer+Collider、manifestのoffsetで顔配置）
using System;
using System.IO;
using System.Reflection;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using GaussianSplatting.Runtime;

public static class FukuwaraiSetup
{
    const string kSourceDir  = "Assets/FaceParts/Source";
    const string kAssetDir   = "Assets/FaceParts/Assets";
    const string kPkgShaders = "Packages/org.nesnausk.gaussian-splatting/Shaders";
    const string kScenePath  = "Assets/Scenes/Fukuwarai.unity";

    [Serializable] class PartCfg { public string name; public float[] offset; public float[] bbox; }
    [Serializable] class SceneCfg { public PartCfg[] parts; }

    [MenuItem("Fukuwarai/セットアップ/グラフィックスAPIを DX12 に (要エディタ再起動)", false, 100)]
    public static void SetDX12()
    {
        var target = BuildTarget.StandaloneWindows64;
        PlayerSettings.SetUseDefaultGraphicsAPIs(target, false);
        PlayerSettings.SetGraphicsAPIs(target, new[] { GraphicsDeviceType.Direct3D12 });
        AssetDatabase.SaveAssets();
        Debug.Log("[Fukuwarai] Standalone graphics API = Direct3D12. エディタを再起動してください（GSはDX12/Vulkan必須）。");
        EditorUtility.DisplayDialog("Fukuwarai",
            "グラフィックスAPIを DX12 に設定しました。\nGaussian Splatting は DX12/Vulkan 必須です。\nUnity エディタを一度再起動してから 手順2 に進んでください。", "OK");
    }

    [MenuItem("Fukuwarai/詳細(3DGS)/GSアセットだけ生成 (.ply→asset)", false, 120)]
    public static void CreateAssets()
    {
        Directory.CreateDirectory(kAssetDir);
        Type creatorType = FindType("GaussianSplatting.Editor.GaussianSplatAssetCreator");
        if (creatorType == null)
        {
            Debug.LogError("[Fukuwarai] GaussianSplatAssetCreator が見つかりません。手動生成(Tools>Gaussian Splats)にフォールバックしてください。");
            return;
        }
        MethodInfo create = creatorType.GetMethod("CreateAsset", BindingFlags.NonPublic | BindingFlags.Instance);
        foreach (var p in LoadConfig().parts)
        {
            string ply = Path.GetFullPath($"{kSourceDir}/{p.name}.ply");
            if (!File.Exists(ply)) { Debug.LogWarning($"[Fukuwarai] not found: {ply}"); continue; }
            try
            {
                var win = ScriptableObject.CreateInstance(creatorType);
                // 最高品質（全 Float32・無圧縮）を強制: m_Quality=VeryHigh(0) → ApplyQualityLevel()
                var qf = creatorType.GetField("m_Quality", BindingFlags.NonPublic | BindingFlags.Instance);
                if (qf != null) qf.SetValue(win, Enum.ToObject(qf.FieldType, 0)); // VeryHigh
                var applyMi = creatorType.GetMethod("ApplyQualityLevel", BindingFlags.NonPublic | BindingFlags.Instance);
                if (applyMi != null) applyMi.Invoke(win, null);
                SetField(win, "m_InputFile", ply);
                SetField(win, "m_OutputFolder", kAssetDir);
                SetField(win, "m_ImportCameras", false);
                create.Invoke(win, null);
                UnityEngine.Object.DestroyImmediate(win);
                Debug.Log($"[Fukuwarai] created splat asset: {p.name}");
            }
            catch (Exception e) { Debug.LogError($"[Fukuwarai] failed {p.name}: {e.Message}"); }
        }
        EditorUtility.ClearProgressBar();
        AssetDatabase.Refresh();
    }

    [MenuItem("Fukuwarai/詳細(3DGS)/GSシーンだけ構築", false, 121)]
    public static void BuildScene()
    {
        var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);

        var shaderSplats    = Load<Shader>($"{kPkgShaders}/RenderGaussianSplats.shader");
        var shaderComposite = Load<Shader>($"{kPkgShaders}/GaussianComposite.shader");
        var shaderPoints    = Load<Shader>($"{kPkgShaders}/GaussianDebugRenderPoints.shader");
        var shaderBoxes     = Load<Shader>($"{kPkgShaders}/GaussianDebugRenderBoxes.shader");
        var csUtil          = Load<ComputeShader>($"{kPkgShaders}/SplatUtilities.compute");

        var root = new GameObject("Face");
        int ok = 0;
        foreach (var p in LoadConfig().parts)
        {
            var go = new GameObject(p.name);
            go.transform.SetParent(root.transform, false);
            go.transform.localPosition = new Vector3(p.offset[0], p.offset[1], p.offset[2]);

            var r = go.AddComponent<GaussianSplatRenderer>();
            var asset = AssetDatabase.LoadAssetAtPath<GaussianSplatAsset>($"{kAssetDir}/{p.name}.asset");
            r.m_Asset = asset;
            r.m_ShaderSplats = shaderSplats;
            r.m_ShaderComposite = shaderComposite;
            r.m_ShaderDebugPoints = shaderPoints;
            r.m_ShaderDebugBoxes = shaderBoxes;
            r.m_CSSplatUtilities = csUtil;

            var col = go.AddComponent<BoxCollider>();
            if (p.bbox != null && p.bbox.Length == 3)
                col.size = new Vector3(p.bbox[0], p.bbox[1], p.bbox[2]);

            if (asset == null) Debug.LogWarning($"[Fukuwarai] {p.name}: GaussianSplatAsset 未生成。先に手順2を実行してください。");
            else ok++;
        }

        var cam = Camera.main;
        if (cam != null)
        {
            cam.transform.position = new Vector3(0f, -0.5f, -2.0f);
            cam.transform.LookAt(new Vector3(0f, -0.5f, 0f));
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.5f, 0.5f, 0.5f);
        }

        Directory.CreateDirectory("Assets/Scenes");
        EditorSceneManager.SaveScene(scene, kScenePath);
        Debug.Log($"[Fukuwarai] built scene '{kScenePath}' (assets assigned: {ok}/4). " +
                  "顔の向きが横/逆なら Face の Transform Rotation を調整してください。");
    }

    [MenuItem("Fukuwarai/★ 3DGS版シーンを生成（立体・推奨）", false, 0)]
    public static void BuildAll() { CreateAssets(); BuildScene(); }

    // ===== テクスチャメッシュ版（obj+png、Unity標準）=====
    const string kMeshDir = "Assets/FaceParts/Meshes";
    const string kTexScenePath = "Assets/Scenes/FukuwaraiTextured.unity";

    // 旧: OBJインポート版（座標/UVの取り込み不具合あり・非推奨）。現行は FukuwaraiMeshBuilder を使用。
    [MenuItem("Fukuwarai/詳細(旧)/OBJ版シーン (非推奨)", false, 200)]
    public static void BuildTexturedScene()
    {
        var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
        var cfg = JsonUtility.FromJson<SceneCfg>(
            AssetDatabase.LoadAssetAtPath<TextAsset>($"{kMeshDir}/textured_scene_config.json").text);

        var shader = Shader.Find("Fukuwarai/UnlitTexturedDoubleSided");
        if (shader == null) Debug.LogError("[Fukuwarai] shader 'Fukuwarai/UnlitTexturedDoubleSided' が見つかりません。");

        var root = new GameObject("Face");
        int ok = 0;
        foreach (var p in cfg.parts)
        {
            string objPath = $"{kMeshDir}/{p.name}/{p.name}.obj";
            var prefab = AssetDatabase.LoadAssetAtPath<GameObject>(objPath);
            if (prefab == null) { Debug.LogWarning($"[Fukuwarai] obj not imported yet: {objPath}"); continue; }
            var go = (GameObject)PrefabUtility.InstantiatePrefab(prefab);
            go.name = p.name;
            go.transform.SetParent(root.transform, false);
            go.transform.localPosition = new Vector3(p.offset[0], p.offset[1], p.offset[2]);

            // 実写テクスチャ + 両面アンリット マテリアルを作成・割当
            var tex = AssetDatabase.LoadAssetAtPath<Texture2D>($"{kMeshDir}/{p.name}/{p.name}_0.png");
            if (shader != null)
            {
                var mat = new Material(shader) { name = p.name + "_mat" };
                if (tex != null) mat.mainTexture = tex;
                string matPath = $"{kMeshDir}/{p.name}/{p.name}_mat.mat";
                AssetDatabase.CreateAsset(mat, matPath);
                foreach (var mr in go.GetComponentsInChildren<MeshRenderer>())
                    mr.sharedMaterial = mat;
                if (tex == null) Debug.LogWarning($"[Fukuwarai] texture not found for {p.name}");
            }

            var col = go.AddComponent<BoxCollider>();
            if (p.bbox != null && p.bbox.Length == 3)
                col.size = new Vector3(p.bbox[0], p.bbox[1], p.bbox[2]);
            ok++;
        }

        var cam = Camera.main;
        if (cam != null)
        {
            cam.transform.position = new Vector3(0f, -0.5f, -1.2f);
            cam.transform.LookAt(new Vector3(0f, -0.5f, 0f));
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.5f, 0.5f, 0.5f);
        }
        var light = UnityEngine.Object.FindObjectOfType<Light>();
        if (light != null) { light.transform.rotation = Quaternion.Euler(20f, 180f, 0f); light.intensity = 1.1f; }

        Directory.CreateDirectory("Assets/Scenes");
        EditorSceneManager.SaveScene(scene, kTexScenePath);
        Debug.Log($"[Fukuwarai] built textured scene '{kTexScenePath}' (parts: {ok}/4). " +
                  "顔の向き/左右が変なら Face の Rotation/Scale を調整。テクスチャが出ない場合は各obj の材質確認。");
    }

    // ---- helpers ----
    static SceneCfg LoadConfig()
    {
        string path = $"{kSourceDir}/scene_config.json";
        var ta = AssetDatabase.LoadAssetAtPath<TextAsset>(path);
        string json = ta != null ? ta.text : File.ReadAllText(Path.GetFullPath(path));
        return JsonUtility.FromJson<SceneCfg>(json);
    }

    static T Load<T>(string path) where T : UnityEngine.Object
    {
        var a = AssetDatabase.LoadAssetAtPath<T>(path);
        if (a == null) Debug.LogWarning($"[Fukuwarai] load failed: {path}");
        return a;
    }

    static Type FindType(string fullName)
    {
        foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
        {
            var t = asm.GetType(fullName);
            if (t != null) return t;
        }
        return null;
    }

    static void SetField(object obj, string field, object value)
    {
        var f = obj.GetType().GetField(field, BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Public);
        if (f == null) throw new Exception($"field '{field}' not found on {obj.GetType().Name}");
        f.SetValue(obj, value);
    }
}
