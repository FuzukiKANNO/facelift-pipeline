// OBJ インポータを介さず、meshdata JSON から Unity 内でメッシュを直接生成する。
// 座標・UV を完全制御できるので「口メッシュ→口の写真」「直立・正しい配置」を保証。
// 材質は両面アンリット(Fukuwarai/UnlitTexturedDoubleSided)＋実写テクスチャ。
using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

public static class FukuwaraiMeshBuilder
{
    const string kDir = "Assets/FaceParts/MeshData";
    const string kScene = "Assets/Scenes/FukuwaraiFace.unity";
    static readonly string[] kParts = { "nose", "mouth", "eyebrow_eye_left", "eyebrow_eye_right" };

    [Serializable] class MeshData { public float[] verts; public float[] uvs; public int[] tris; public float[] alpha; public float[] offset; public float[] bbox; }

    [MenuItem("Fukuwarai/★ 写真テクスチャ版シーンを生成（メッシュ）", false, 1)]
    public static void Build()
    {
        var shader = Shader.Find("Fukuwarai/UnlitTexturedFade");
        if (shader == null) shader = Shader.Find("Fukuwarai/UnlitTexturedDoubleSided");
        if (shader == null) { Debug.LogError("[Fukuwarai] Fukuwarai シェーダが見つかりません"); return; }

        var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
        var root = new GameObject("Face");
        int ok = 0;
        var center = Vector3.zero; float maxExtent = 0.1f;
        foreach (var name in kParts)
        {
            string jsonPath = $"{kDir}/{name}/{name}.meshdata.json";
            var ta = AssetDatabase.LoadAssetAtPath<TextAsset>(jsonPath);
            if (ta == null) { Debug.LogWarning($"[Fukuwarai] missing {jsonPath}"); continue; }
            var md = JsonUtility.FromJson<MeshData>(ta.text);

            var mesh = new Mesh { indexFormat = IndexFormat.UInt32 };
            var vs = new Vector3[md.verts.Length / 3];
            for (int i = 0; i < vs.Length; i++) vs[i] = new Vector3(md.verts[3*i], md.verts[3*i+1], md.verts[3*i+2]);
            var uv = new Vector2[md.uvs.Length / 2];
            for (int i = 0; i < uv.Length; i++) uv[i] = new Vector2(md.uvs[2*i], md.uvs[2*i+1]);
            mesh.vertices = vs; mesh.uv = uv; mesh.triangles = md.tris;
            if (md.alpha != null && md.alpha.Length == vs.Length)
            {
                var cols = new Color[vs.Length];
                for (int i = 0; i < cols.Length; i++) cols[i] = new Color(1f, 1f, 1f, md.alpha[i]);
                mesh.colors = cols;   // 頂点アルファ（縁フェード）
            }
            mesh.RecalculateNormals(); mesh.RecalculateBounds();
            AssetDatabase.CreateAsset(mesh, $"{kDir}/{name}/{name}_mesh.asset");

            var go = new GameObject(name);
            go.transform.SetParent(root.transform, false);
            var pos = new Vector3(md.offset[0], md.offset[1], md.offset[2]);
            go.transform.localPosition = pos;
            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            var mr = go.AddComponent<MeshRenderer>();

            var tex = AssetDatabase.LoadAssetAtPath<Texture2D>($"{kDir}/{name}/{name}_tex.png");
            var mat = new Material(shader) { name = name + "_mat" };
            if (tex != null) mat.mainTexture = tex;
            AssetDatabase.CreateAsset(mat, $"{kDir}/{name}/{name}_mat.mat");
            mr.sharedMaterial = mat;

            var col = go.AddComponent<BoxCollider>();
            if (md.bbox != null && md.bbox.Length == 3) col.size = new Vector3(md.bbox[0], md.bbox[1], md.bbox[2]);

            center += pos; maxExtent = Mathf.Max(maxExtent, pos.magnitude); ok++;
        }
        if (ok > 0) center /= ok;

        var cam = Camera.main;
        if (cam != null)
        {
            float dist = Mathf.Max(0.8f, maxExtent * 3f);
            cam.transform.position = center + new Vector3(0, 0, -dist);
            cam.transform.LookAt(center);
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.5f, 0.5f, 0.5f);
            cam.nearClipPlane = 0.01f;
        }

        Directory.CreateDirectory("Assets/Scenes");
        EditorSceneManager.SaveScene(scene, kScene);
        AssetDatabase.SaveAssets();
        Debug.Log($"[Fukuwarai] MeshData face built ({ok}/4) -> {kScene}. " +
                  "向きが奥向き/逆なら Face を Y軸で180°回してください。");
    }
}
