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

public class HalconOrientedObjectDetectionDataEnhancement : IHalconDataEnhancement<
    HalconOrientedObjectDetectionDataEnhancement>
{

    public delegate (HImage Image, List<long> BboxLabelId, List<double> BboxRow,
        List<double> BboxCol, List<double> BboxLength1, List<double> BboxLength2,
        List<double> BboxPhi)[] 简单增强委托(HImage image,
            List<long>? bboxLabelId,
            List<double>? bboxRow,
            List<double>? bboxCol,
            List<double>? bboxLength1,
            List<double>? bboxLength2,
            List<double>? bboxPhi);

    private IEnumerable<DataEnhancementImageInfo>? _dataEnhancementImageInfos;

    private HalconOrientedObjectDetectionDict? _sourceDict;

    private IEnumerable<SourceImageInfo> _sourceImageInfos = null!;

    public HalconOrientedObjectDetectionDataEnhancement LoadSouce(HDict hDict)
    {
        数据源不能已加载();
        _sourceDict = new HalconOrientedObjectDetectionDict();
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

                var newDict = new HalconOrientedObjectDetectionDict
                {
                    ImageDir = newImageDir,
                    Names = _sourceDict!.Names,
                    Ids = _sourceDict.Ids,
                    Samples = new List<HalconOrientedObjectDetectionDict.Sample>()
                };

                token ??= CancellationToken.None;

                foreach (var imageInfo in _dataEnhancementImageInfos)
                {
                    token.Value.ThrowIfCancellationRequested();

                    newDict.Samples.Add(new HalconOrientedObjectDetectionDict.Sample
                    {
                        Id = imageInfo.Id,
                        FileName = imageInfo.FileName,
                        BboxLabelId = imageInfo.BboxLabelId,
                        BboxRow = imageInfo.BboxRow,
                        BboxCol = imageInfo.BboxCol,
                        BboxLength1 = imageInfo.BboxLength1,
                        BboxLength2 = imageInfo.BboxLength2,
                        BboxPhi = imageInfo.BboxPhi
                    });

                    var newImagePath = Path.Combine(newImageDir, imageInfo.FileName!);
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

    public HalconOrientedObjectDetectionDataEnhancement DataEnhancement(
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

    public HalconOrientedObjectDetectionDataEnhancement SimpleDataEnhancement(
        简单增强委托 func)
    {
        数据源不能未加载();
        var infos = new List<DataEnhancementImageInfo>(100);
        var count = 0;

        foreach (var sourceImageInfo in _sourceImageInfos)
        {
            var results = func.Invoke(sourceImageInfo.Image,
                sourceImageInfo.BboxLabelId,
                sourceImageInfo.BboxRow,
                sourceImageInfo.BboxCol,
                sourceImageInfo.BboxLength1,
                sourceImageInfo.BboxLength2,
                sourceImageInfo.BboxPhi);

            foreach (
                var (Image, BboxLabelId, BboxRow, BboxCol, BboxLength1, BboxLength2,
                    BboxPhi) in results)
            {
                count++;

                infos.Add(new DataEnhancementImageInfo
                {
                    Image = Image,
                    Id = count,
                    FileName =
                        $"{Path.GetFileNameWithoutExtension(sourceImageInfo.FileName)}_{count}.png",
                    BboxLabelId = BboxLabelId,
                    BboxRow = BboxRow,
                    BboxCol = BboxCol,
                    BboxLength1 = BboxLength1,
                    BboxLength2 = BboxLength2,
                    BboxPhi = BboxPhi
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
                sample.Id!.Value,
                sample.FileName!,
                sample.BboxLabelId!,
                sample.BboxRow!,
                sample.BboxCol!,
                sample.BboxLength1!,
                sample.BboxLength2!,
                sample.BboxPhi!))
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

        public List<double> BboxCol { get; }

        public List<long> BboxLabelId { get; }

        public List<double> BboxLength1 { get; }

        public List<double> BboxLength2 { get; }

        public List<double> BboxPhi { get; }

        public List<double> BboxRow { get; }

    }

    public class DataEnhancementImageInfo : IImageInfo
    {

        public List<double> BboxCol { get; set; } = null!;

        public List<long> BboxLabelId { get; set; } = null!;

        public List<double> BboxLength1 { get; set; } = null!;

        public List<double> BboxLength2 { get; set; } = null!;

        public List<double> BboxPhi { get; set; } = null!;

        public List<double> BboxRow { get; set; } = null!;

        public string FileName { get; set; } = null!;

        public long Id { get; set; }

        public HImage Image { get; set; } = null!;

    }

    public class SourceImageInfo : IImageInfo
    {

        public SourceImageInfo(string imageDir,
            long id,
            string fileName,
            List<long> bboxLabelId,
            List<double> bboxRow,
            List<double> bboxCol,
            List<double> bboxLength1,
            List<double> bboxLength2,
            List<double> bboxPhi)
        {
            ImageDir = imageDir;
            Id = id;
            FileName = fileName;
            BboxLabelId = bboxLabelId;
            BboxRow = bboxRow;
            BboxCol = bboxCol;
            BboxLength1 = bboxLength1;
            BboxLength2 = bboxLength2;
            BboxPhi = bboxPhi;
            var imagePath = Path.Combine(imageDir, fileName!);
            Image = new HImage(imagePath);
        }

        public string ImageDir { get; }

        public List<double> BboxCol { get; }

        public List<long> BboxLabelId { get; }

        public List<double> BboxLength1 { get; }

        public List<double> BboxLength2 { get; }

        public List<double> BboxPhi { get; }

        public List<double> BboxRow { get; }

        public string FileName { get; }

        public long Id { get; }

        public HImage Image { get; }

    }

}