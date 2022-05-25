using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using HalconDotNet;
using Halcon深度学习数据增强.Dicts;

namespace Halcon深度学习数据增强.DataEnhancements;

/// <summary>
/// Halcon实例分割数据增强
/// </summary>
public class HalconInstanceSegmentationDataEnhancement
{

    public delegate (HImage Image, List<HRegion> Mask, List<long> BboxLabelId,
        List<double> BboxRow1, List<double> BboxCol1, List<double> BboxRow2,
        List<double> BboxCol2)[] 简单增强委托(HImage image,
            List<HRegion> mask,
            List<long> bboxLabelId,
            List<double> bboxRow1,
            List<double> bboxCol1,
            List<double> bboxRow2,
            List<double> bboxCol2);

    private IEnumerable<DataEnhancementImageInfo>? _dataEnhancementImageInfos;

    private HalconInstanceSegmentationDict? _sourceDict;

    private IEnumerable<SourceImageInfo> _sourceImageInfos = null!;

    public HalconInstanceSegmentationDataEnhancement DataEnhancement(
        Func<SourceImageInfo, DataEnhancementImageInfo[]> func)
    {
        数据源不能未加载();
        var infos = new List<DataEnhancementImageInfo>(100);

        foreach (var sourceImageInfo in _sourceImageInfos)
        {
            var dataEnhancementImageInfo = func.Invoke(sourceImageInfo);
            infos.AddRange(dataEnhancementImageInfo);
        }

        _dataEnhancementImageInfos = infos;

        return this;
    }

    public HalconInstanceSegmentationDataEnhancement LoadSouce(HDict hDict)
    {
        数据源不能已加载();
        _sourceDict = HalconInstanceSegmentationDict.FromHDict(hDict);
        var errors = _sourceDict.Errors().ToArray();

        if (errors.Any()) throw new Exception(string.Join("\n", errors));

        _sourceImageInfos = 解析数据();

        return this;
    }

    public HalconInstanceSegmentationDataEnhancement LoadSourceFromPath(string dictPath,
        HTuple? genParamName = default,
        HTuple? genParamValue = default)
    {
        数据源不能已加载();
        genParamName ??= new HTuple();
        genParamValue ??= new HTuple();
        var hDict = new HDict(dictPath, genParamName, genParamValue);

        return LoadSouce(hDict);
    }

    public Task Save(string? newImageDir = default,
        string? newDictPath = default,
        HTuple? genParamName = default,
        HTuple? genParamValue = default,
        CancellationToken? token = default)
    {
        return Task.Factory.StartNew(() =>
            {
                if (_dataEnhancementImageInfos == null)

                    //没有设置数据增强方法
                    throw new Exception("没有设置数据增强方法");

                var baseDirectory = AppDomain.CurrentDomain.BaseDirectory;
                newImageDir ??= Path.Combine(baseDirectory, "DataEnhancementImages");

                if (Directory.Exists(newImageDir) is false)
                    Directory.CreateDirectory(newImageDir);

                newDictPath ??=
                    Path.Combine(baseDirectory, "DataEnhancementImagesDict.hdict");

                if (Directory.Exists(baseDirectory) is false)
                    Directory.CreateDirectory(baseDirectory);

                var newDict = new HalconInstanceSegmentationDict
                {
                    ImageDir = newImageDir,
                    Names = _sourceDict!.Names,
                    Ids = _sourceDict.Ids,
                    Samples = new List<HalconInstanceSegmentationDict.Sample>()
                };

                token ??= CancellationToken.None;

                foreach (var imageInfo in _dataEnhancementImageInfos)
                {
                    token.Value.ThrowIfCancellationRequested();

                    newDict.Samples.Add(new HalconInstanceSegmentationDict.Sample
                    {
                        Id = imageInfo.Id,
                        FileName = imageInfo.FileName,
                        Mask = imageInfo.Mask,
                        BboxLabelId = imageInfo.BboxLabelId,
                        BboxRow1 = imageInfo.BboxRow1,
                        BboxCol1 = imageInfo.BboxCol1,
                        BboxRow2 = imageInfo.BboxRow2,
                        BboxCol2 = imageInfo.BboxCol2
                    });

                    var newImagePath = Path.Combine(newImageDir, imageInfo.FileName);
                    var image = imageInfo.Image;

                    var extension = Path.GetExtension(newImagePath)
                        .Replace(".", string.Empty);

                    image.WriteImage(extension, 0, newImagePath);
                }

                genParamName ??= new HTuple();
                genParamValue ??= new HTuple();
                token.Value.ThrowIfCancellationRequested();
                var dict = newDict.ToHDict();
                dict.WriteDict(newDictPath, genParamName, genParamValue);
            },
            TaskCreationOptions.LongRunning);
    }

    public HalconInstanceSegmentationDataEnhancement SimpleDataEnhancement(简单增强委托 func)
    {
        数据源不能未加载();
        var infos = new List<DataEnhancementImageInfo>(100);
        var count = 0;

        foreach (var sourceImageInfo in _sourceImageInfos)
        {
            var results = func.Invoke(sourceImageInfo.Image,
                sourceImageInfo.Mask!,
                sourceImageInfo.BboxLabelId!,
                sourceImageInfo.BboxRow1!,
                sourceImageInfo.BboxCol1!,
                sourceImageInfo.BboxRow2!,
                sourceImageInfo.BboxCol2!);

            foreach (var r in results)
            {
                count++;

                infos.Add(new DataEnhancementImageInfo
                {
                    Image = r.Image,
                    Id = count,
                    FileName =
                        $"{Path.GetFileNameWithoutExtension(sourceImageInfo.FileName)}_{count}.png",
                    Mask = r.Mask,
                    BboxLabelId = r.BboxLabelId,
                    BboxRow1 = r.BboxRow1,
                    BboxCol1 = r.BboxCol1,
                    BboxRow2 = r.BboxRow2,
                    BboxCol2 = r.BboxCol2
                });
            }
        }

        _dataEnhancementImageInfos = infos;

        return this;
    }

    private IEnumerable<SourceImageInfo> 解析数据()
    {
        var samples = _sourceDict!.Samples!;

        return samples.Select(sample => new SourceImageInfo(_sourceDict.ImageDir,
                sample.Id,
                sample.FileName,
                sample.Mask,
                sample.BboxLabelId,
                sample.BboxRow1,
                sample.BboxCol1,
                sample.BboxRow2,
                sample.BboxCol2))
            .ToArray();
    }

    private void 数据源不能未加载()
    {
        if (_sourceDict == null) throw new Exception("数据源未加载");
    }

    private void 数据源不能已加载()
    {
        if (_sourceDict != null) throw new Exception("数据已经加载");
    }

    public class DataEnhancementImageInfo
    {

        public List<double>? BboxCol1 { get; set; }

        public List<double>? BboxCol2 { get; set; }

        public List<long>? BboxLabelId { get; set; }

        public List<double>? BboxRow1 { get; set; }

        public List<double>? BboxRow2 { get; set; }

        public string? FileName { get; set; }

        public long? Id { get; set; }

        public HImage Image { get; set; }

        public List<HRegion>? Mask { get; set; }

    }

    public class SourceImageInfo
    {

        public SourceImageInfo(string? imageDir,
            long? id,
            string? fileName,
            List<HRegion>? mask,
            List<long>? bboxLabelId,
            List<double>? bboxRow1,
            List<double>? bboxCol1,
            List<double>? bboxRow2,
            List<double>? bboxCol2)
        {
            ImageDir = imageDir;
            Id = id;
            FileName = fileName;
            Mask = mask;
            BboxLabelId = bboxLabelId;
            BboxRow1 = bboxRow1;
            BboxCol1 = bboxCol1;
            BboxRow2 = bboxRow2;
            BboxCol2 = bboxCol2;
            var path = Path.Combine(imageDir!, fileName!);
            Image = new HImage(path);
        }

        public List<double>? BboxCol1 { get; }

        public List<double>? BboxCol2 { get; }

        public List<long>? BboxLabelId { get; }

        public List<double>? BboxRow1 { get; }

        public List<double>? BboxRow2 { get; }

        public string? FileName { get; }

        public long? Id { get; }

        public HImage Image { get; }

        public string? ImageDir { get; }

        public List<HRegion>? Mask { get; }

    }

}