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

/// <summary>
///     Halcon分类数据增强
/// </summary>
public class
    HalconClassificationDataEnhancement : IHalconDataEnhancement<
        HalconClassificationDataEnhancement>
{

    public delegate HImage[] 简单增强委托(HImage sourceImage);

    private IEnumerable<DataEnhancementImageInfo>? _dataEnhancementImageInfos;

    private HalconClassificationDict? _sourceDict;

    private IEnumerable<SourceImageInfo> _sourceImageInfos = null!;

    public HalconClassificationDataEnhancement LoadSouce(HDict hDict)
    {
        数据源不能已加载();
        _sourceDict = new HalconClassificationDict();
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

                var newDict = new HalconClassificationDict
                {
                    ImageDir = newImageDir,
                    Names = _sourceDict!.Names,
                    Ids = _sourceDict.Ids,
                    Samples = new List<HalconClassificationDict.Sample>()
                };

                token ??= CancellationToken.None;

                foreach (var imageInfo in _dataEnhancementImageInfos)
                {
                    token.Value.ThrowIfCancellationRequested();

                    newDict.Samples.Add(new HalconClassificationDict.Sample
                    {
                        Id = imageInfo.Id,
                        FileName = imageInfo.FileName,
                        LabelId = imageInfo.LabelId
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

    public HalconClassificationDataEnhancement DataEnhancement(
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

    public HalconClassificationDataEnhancement SimpleDataEnhancement(简单增强委托 func)
    {
        数据源不能未加载();
        var infos = new List<DataEnhancementImageInfo>(100);
        var count = 0;

        foreach (var sourceImageInfo in _sourceImageInfos)
        {
            var images = func.Invoke(sourceImageInfo.Image);

            foreach (var image in images)
            {
                count++;

                infos.Add(new DataEnhancementImageInfo
                {
                    Image = image,
                    Id = count,
                    LabelId = sourceImageInfo.LabelId,
                    FileName =
                        $"{Path.GetFileNameWithoutExtension(sourceImageInfo.FileName)}_{count}.png"
                });
            }
        }

        _dataEnhancementImageInfos = infos;

        return this;
    }

    private IEnumerable<SourceImageInfo> 解析数据()
    {
        var samples = _sourceDict!.Samples!;

        return samples.Select(sample => new SourceImageInfo(_sourceDict!.ImageDir!,
                sample.Id!.Value,
                sample.FileName!,
                sample.LabelId!.Value))
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

        public long LabelId { get; }

    }

    public class DataEnhancementImageInfo : IImageInfo
    {

        public string FileName { get; set; } = null!;

        public long Id { get; set; }

        public HImage Image { get; set; } = null!;

        public long LabelId { get; set; }

    }

    public class SourceImageInfo : IImageInfo
    {

        public SourceImageInfo(string imageDir, long id, string fileName, long labelId)
        {
            ImageDir = imageDir;
            Id = id;
            FileName = fileName;
            LabelId = labelId;
            var imagePath = Path.Combine(imageDir, fileName);
            Image = new HImage(imagePath);
        }

        public string ImageDir { get; }

        public string FileName { get; }

        public long Id { get; }

        public HImage Image { get; }

        public long LabelId { get; }

    }

}

public interface IHalconDataEnhancement<out TImplement>
{

    TImplement LoadSouce(HDict hDict);

    Task Save(string? newImageDir = null,
        string? newDictPath = null,
        HTuple? genParamName = null,
        HTuple? genParamValue = null,
        CancellationToken? token = null);

}