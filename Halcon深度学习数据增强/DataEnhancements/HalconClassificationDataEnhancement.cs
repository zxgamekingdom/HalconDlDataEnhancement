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
///     Halcon分类数据增强
/// </summary>
public class HalconClassificationDataEnhancement
{

    private IEnumerable<DataEnhancementImageInfo>? _dataEnhancementImageInfos;

    private HalconClassificationDict? _sourceDict;

    private IEnumerable<SourceImageInfo> _sourceImageInfos = null!;

    public HalconClassificationDataEnhancement LoadSouce(HDict hDict)
    {
        数据源不能已加载();
        _sourceDict = HalconClassificationDict.FromHDict(hDict);
        var errors = _sourceDict.Errors().ToArray();

        if (errors.Any()) throw new Exception(string.Join("\n", errors));

        _sourceImageInfos = 解析数据();

        return this;
    }

    private IEnumerable<SourceImageInfo> 解析数据()
    {
        var samples = _sourceDict!.Samples!;

        return samples.Select(sample => new SourceImageInfo(sample.Id!.Value,
                _sourceDict!.ImageDir!,
                sample.FileName!,
                sample.LabelId!.Value))
            .ToArray();
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

    private void 数据源不能未加载()
    {
        if (_sourceDict == null) throw new Exception("数据源未加载");
    }

    public HalconClassificationDataEnhancement SimpleDataEnhancement(
        Func<HImage, HImage[]> func)
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

    private void 数据源不能已加载()
    {
        if (_sourceDict != null) throw new Exception("数据已经加载");
    }

    public HalconClassificationDataEnhancement LoadSourceFromPath(string dictPath,
        HTuple? genParamName = default,
        HTuple? genParamValue = default)
    {
        数据源不能已加载();
        genParamName ??= new HTuple();
        genParamValue ??= new HTuple();
        var hDict = new HDict(dictPath, genParamName, genParamValue);

        return LoadSouce(hDict);
    }

    public class DataEnhancementImageInfo
    {

        public HImage Image { get; set; }

        public long Id { get; set; }

        public long LabelId { get; set; }

        public string FileName { get; set; }

    }

    public class SourceImageInfo
    {

        public SourceImageInfo(long id, string imageDir, string fileName, long labelId)
        {
            Id = id;
            Dir = imageDir;
            FileName = fileName;
            LabelId = labelId;
            var path = Path.Combine(Dir, FileName);
            Image = new HImage(path);
        }

        public HImage Image { get; }

        public long LabelId { get; }

        public string FileName { get; }

        public string Dir { get; }

        public long Id { get; }

    }

}
