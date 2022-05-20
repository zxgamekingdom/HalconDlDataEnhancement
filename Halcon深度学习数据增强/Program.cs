using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using HalconDotNet;
using Halcon深度学习数据增强.Dicts;
using Halcon深度学习数据增强.Tools;
using static HalconDotNet.HOperatorSet;

namespace Halcon深度学习数据增强;

public static class Program
{

    [STAThread]
    public static void Main(string[] args)
    {
        HalconEnvironmentInit.Init();
        var list = 获取();
        处理(list);
    }

    private static List<(string, string)> 获取()
    {
        var path = @"C:\Users\Zhou Taurus\Desktop\新建文件夹 (2)\FTT.hdict";

        var dict =
            HalconSemanticSegmentationDict.FromHDict(new HDict(path,
                new HTuple(),
                new HTuple()));

        var list = new List<(string, string)>();

        foreach (var sample in dict.Samples)
        {
            var fileName = sample.SegmentationFileName;
            var dir = dict.SegmentationDir;
            var s = Path.Combine(dir, fileName);
            ReadImage(out var image, s);
            MinMaxGray(image, image, 0, out var min, out var max, out var range);

            if (max.D >= 1)
                list.Add((Path.Combine(dict.ImageDir, sample.FileName),
                    Path.Combine(dict.SegmentationDir, sample.SegmentationFileName)));
        }

        return list;
    }

    private static void 处理(List<(string, string)> list)
    {
        var newDir = @"C:\Users\Zhou Taurus\Desktop\新建文件夹 (2)\新建文件夹";
        var sourceImageDir = Path.Combine(newDir, "source");
        var maskImageDir = Path.Combine(newDir, "mask");
        if (Directory.Exists(sourceImageDir)) Directory.Delete(sourceImageDir, true);
        if (Directory.Exists(maskImageDir)) Directory.Delete(maskImageDir, true);
        Directory.CreateDirectory(sourceImageDir);
        Directory.CreateDirectory(maskImageDir);
        var images = new List<(HObject, HObject)>();

        foreach (var (sourcePath, maskPath) in list)
        {
            ReadImage(out var sourceImage, sourcePath);
            ReadImage(out var maskImage, maskPath);
            Threshold(maskImage, out var region, 1, 255);
            ReduceDomain(sourceImage, region, out var imageReduced);
            images.AddRange(增强(imageReduced, maskImage));
        }

        var count = 0;

        foreach (var (item1, item2) in images)
        {
            WriteImage(item1, "png", 0, Path.Combine(sourceImageDir, $"{count}.png"));
            WriteImage(item2, "png", 0, Path.Combine(maskImageDir, $"{count}.png"));
            count++;

            if (count == 10) break;
        }

        Process.Start("explorer.exe", sourceImageDir);
    }

    private static IEnumerable<(HObject, HObject)> 增强(HObject sourceImage,
        HObject maskImage)
    {
        var list = new List<(HObject, HObject)>();
        RotateImage(sourceImage, out var sourceRotate, 180, "weighted");
        RotateImage(maskImage, out var maskRotate, 180, "weighted");

        (HObject, HObject)[] images =
        {
            (sourceImage, maskImage), (sourceRotate, maskRotate)
        };

        foreach (var (item1, item2) in images)
        {
            Threshold(item2, out var region, 1, 255);
            ReduceDomain(item1, region, out var imageReduced);
            list.Add((imageReduced, item2));
            Illuminate(imageReduced, out var illuminate, 100, 100, .7);
            list.Add((illuminate, item2));
            Emphasize(imageReduced, out var emphasize, 11, 11, 1);
            list.Add((emphasize, item2));
            MeanImage(imageReduced, out var mean, 4, 4);
            list.Add((mean, item2));
        }

        return list;
    }

}
