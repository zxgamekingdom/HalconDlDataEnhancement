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

public class
    HalconObjectDetectionDataEnhancement : IHalconDataEnhancement<
        HalconObjectDetectionDataEnhancement>
{

    public delegate (HImage Image, List<long> BboxLabelId, List<double> BboxRow1,
        List<double> BboxCol1, List<double> BboxRow2, List<double> BboxCol2)[] 简单增强委托(
            HImage image,
            List<long> bboxLabelId,
            List<double> bboxRow1,
            List<double> bboxCol1,
            List<double> bboxRow2,
            List<double> bboxCol2);

    private IEnumerable<DataEnhancementImageInfo>? _dataEnhancementImageInfos;

    private HalconObjectDetectionDict? _sourceDict;

    private IEnumerable<SourceImageInfo> _sourceImageInfos = null!;

    public HalconObjectDetectionDataEnhancement LoadSouce(HDict hDict)
    {
        数据源不能已加载();
        _sourceDict = new HalconObjectDetectionDict();
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

                var newDict = new HalconObjectDetectionDict
                {
                    ImageDir = newImageDir,
                    Names = _sourceDict!.Names,
                    Ids = _sourceDict.Ids,
                    Samples = new List<HalconObjectDetectionDict.Sample>()
                };

                token ??= CancellationToken.None;

                foreach (var imageInfo in _dataEnhancementImageInfos)
                {
                    token.Value.ThrowIfCancellationRequested();

                    newDict.Samples.Add(new HalconObjectDetectionDict.Sample
                    {
                        Id = imageInfo.Id,
                        FileName = imageInfo.FileName,
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

    public HalconObjectDetectionDataEnhancement DataEnhancement(
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

    public HalconObjectDetectionDataEnhancement SimpleDataEnhancement(简单增强委托 func)
    {
        数据源不能未加载();
        var infos = new List<DataEnhancementImageInfo>(100);
        var count = 0;

        foreach (var sourceImageInfo in _sourceImageInfos)
        {
            var results = func.Invoke(sourceImageInfo.Image,
                sourceImageInfo.BboxLabelId,
                sourceImageInfo.BboxRow1,
                sourceImageInfo.BboxCol1,
                sourceImageInfo.BboxRow2,
                sourceImageInfo.BboxCol2);

            foreach (var (image, labelId, row1, col1, row2, col2) in results)
            {
                count++;

                infos.Add(new DataEnhancementImageInfo
                {
                    Image = image,
                    Id = count,
                    FileName =
                        $"{Path.GetFileNameWithoutExtension(sourceImageInfo.FileName)}_{count}.png",
                    BboxLabelId = labelId,
                    BboxRow1 = row1,
                    BboxCol1 = col1,
                    BboxRow2 = row2,
                    BboxCol2 = col2
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
                sample.BboxLabelId!,
                sample.BboxRow1!,
                sample.BboxCol1!,
                sample.BboxRow2!,
                sample.BboxCol2!))
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

        public List<double> BboxCol1 { get; }

        public List<double> BboxCol2 { get; }

        public List<long> BboxLabelId { get; }

        public List<double> BboxRow1 { get; }

        public List<double> BboxRow2 { get; }

    }

    public class DataEnhancementImageInfo : IImageInfo
    {

        public long Id { get; set; }

        public List<double> BboxCol1 { get; set; } = null!;

        public List<double> BboxCol2 { get; set; } = null!;

        public List<long> BboxLabelId { get; set; } = null!;

        public List<double> BboxRow1 { get; set; } = null!;

        public List<double> BboxRow2 { get; set; } = null!;

        public string FileName { get; set; } = null!;

        public HImage Image { get; set; } = null!;

    }

    public class SourceImageInfo : IImageInfo
    {

        public SourceImageInfo(string imageDir,
            long id,
            string fileName,
            List<long> bboxLabelId,
            List<double> bboxRow1,
            List<double> bboxCol1,
            List<double> bboxRow2,
            List<double> bboxCol2)
        {
            ImageDir = imageDir;
            Id = id;
            FileName = fileName;
            BboxLabelId = bboxLabelId;
            BboxRow1 = bboxRow1;
            BboxCol1 = bboxCol1;
            BboxRow2 = bboxRow2;
            BboxCol2 = bboxCol2;
            var imagePath = Path.Combine(imageDir, fileName);
            Image = new HImage(imagePath);
        }

        public string ImageDir { get; }

        public long Id { get; }

        public List<double> BboxCol1 { get; }

        public List<double> BboxCol2 { get; }

        public List<long> BboxLabelId { get; }

        public List<double> BboxRow1 { get; }

        public List<double> BboxRow2 { get; }

        public string FileName { get; }

        public HImage Image { get; }

    }

}
