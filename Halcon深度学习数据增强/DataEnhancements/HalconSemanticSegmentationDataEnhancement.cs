using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using HalconDotNet;
using Halcon深度学习数据增强.DataEnhancements.Abstracts;
using Halcon深度学习数据增强.Dicts;

namespace Halcon深度学习数据增强.DataEnhancements;

public class HalconSemanticSegmentationDataEnhancement : IHalconDataEnhancement<
    HalconSemanticSegmentationDataEnhancement>
{

    public delegate (HImage Image, HImage SegmentationImage)[] 简单增强委托(HImage image,
        HImage segmentationImage);

    private IEnumerable<DataEnhancementImageInfo>? _dataEnhancementImageInfos;

    private HalconSemanticSegmentationDict? _sourceDict;

    private IEnumerable<SourceImageInfo> _sourceImageInfos = null!;

    public HalconSemanticSegmentationDataEnhancement LoadSouce(HDict hDict)
    {
        数据源不能已加载();
        _sourceDict = new HalconSemanticSegmentationDict();
        _sourceDict.FromHDict(hDict);
        var errors = _sourceDict.Errors().ToArray();

        if (errors.Length > 0) throw new Exception(string.Join("\n", errors));

        _sourceImageInfos = 解析数据();

        return this;
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

                var newDict = new HalconSemanticSegmentationDict
                {
                    ImageDir = newImageDir,
                    Names = _sourceDict!.Names,
                    Ids = _sourceDict.Ids,
                    Samples = new List<HalconSemanticSegmentationDict.Sample>()
                };

                token ??= CancellationToken.None;

                foreach (var imageInfo in _dataEnhancementImageInfos)
                {
                    token.Value.ThrowIfCancellationRequested();

                    newDict.Samples.Add(new HalconSemanticSegmentationDict.Sample
                    {
                        Id = imageInfo.Id,
                        FileName = imageInfo.FileName,
                        SegmentationFileName = imageInfo.SegmentationFileName
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

    public HalconSemanticSegmentationDataEnhancement DataEnhancement(
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

    public HalconSemanticSegmentationDataEnhancement SimpleDataEnhancement(简单增强委托 func)
    {
        数据源不能未加载();
        var infos = new List<DataEnhancementImageInfo>(100);
        var count = 0;

        foreach (var sourceImageInfo in _sourceImageInfos)
        {
            var images = func.Invoke(sourceImageInfo.Image,
                sourceImageInfo.SegmentationImage);

            foreach (var (image, segmentationImage) in images)
            {
                count++;

                infos.Add(new DataEnhancementImageInfo
                {
                    Id = count,
                    Image = image,
                    SegmentationImage = segmentationImage,
                    FileName =
                        $"{Path.GetFileNameWithoutExtension(sourceImageInfo.FileName)}_{count}.png",
                    SegmentationFileName =
                        $"{Path.GetFileNameWithoutExtension(sourceImageInfo.SegmentationFileName)}_{count}.png"
                });
            }
        }

        _dataEnhancementImageInfos = infos;

        return this;
    }

    private IEnumerable<SourceImageInfo> 解析数据()
    {
        var samples = _sourceDict!.Samples!;

        return samples.Select(sample => new SourceImageInfo(_sourceDict.ImageDir!,
                _sourceDict.SegmentationDir!,
                sample.Id!.Value,
                sample.FileName!,
                sample.SegmentationFileName!))
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

    public interface IImageInfo : IDataEnhancementImageInfo
    {

        public string SegmentationFileName { get; }

        public HImage SegmentationImage { get; }

    }

    public class DataEnhancementImageInfo
    {

        public string FileName { get; set; } = null!;

        public long Id { get; set; }

        public HImage Image { get; set; } = null!;

        public string SegmentationFileName { get; set; } = null!;

        public HImage SegmentationImage { get; set; } = null!;

    }

    public class SourceImageInfo : IImageInfo
    {

        public SourceImageInfo(string imageDir,
            string segmentationDir,
            long id,
            string fileName,
            string segmentationFileName)
        {
            ImageDir = imageDir;
            SegmentationDir = segmentationDir;
            Id = id;
            FileName = fileName;
            SegmentationFileName = segmentationFileName;
            var imagePath = Path.Combine(imageDir, fileName);
            Image = new HImage(imagePath);

            var segmentationImagePath =
                Path.Combine(segmentationDir, segmentationFileName);

            SegmentationImage = new HImage(segmentationImagePath);
        }

        public string ImageDir { get; }

        public string SegmentationDir { get; }

        public string FileName { get; }

        public long Id { get; }

        public HImage Image { get; }

        public string SegmentationFileName { get; }

        public HImage SegmentationImage { get; }

    }

}
