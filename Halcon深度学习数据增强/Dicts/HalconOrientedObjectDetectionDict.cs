using System.Collections.Generic;
using System.Linq;
using HalconDotNet;
using Halcon深度学习数据增强.Abstracts;
using Halcon深度学习数据增强.Dicts.Abstracts;
using Halcon深度学习数据增强.Dicts.Extensions;
using Halcon深度学习数据增强.Dicts.Tools;

namespace Halcon深度学习数据增强.Dicts;

public class
    HalconOrientedObjectDetectionDict : IHalconDict<
        HalconOrientedObjectDetectionDict.Sample>
{

    public List<long>? Ids { get; set; }

    public string? ImageDir { get; set; }

    public List<string>? Names { get; set; }

    public List<Sample>? Samples { get; set; }

    public void FromHDict(HDict dict)
    {
        Ids = dict.FromKeyTuple("class_ids", x => x.LArr.ToList());
        Names = dict.FromKeyTuple("class_names", x => x.SArr.ToList());
        ImageDir = dict.FromKeyTuple("image_dir", x => x.S);
        var handles = dict.FromKeyTuple("samples", x => x.HArr);

        if (handles == null) return;

        var samplesLength = handles.Length;
        var list = new List<Sample>(samplesLength + 10);

        for (var i = 0; i < samplesLength; i++)
        {
            var d = new HDict(handles[i]);

            list.Add(new Sample
            {
                Id = d.FromKeyTuple("image_id", x => x.L),
                FileName = d.FromKeyTuple("image_file_name", x => x.S),
                BboxLabelId = d.FromKeyTuple("bbox_label_id", x => x.LArr.ToList()),
                BboxRow = d.FromKeyTuple("bbox_row", x => x.DArr.ToList()),
                BboxCol = d.FromKeyTuple("bbox_col", x => x.DArr.ToList()),
                BboxLength1 = d.FromKeyTuple("bbox_length1", x => x.DArr.ToList()),
                BboxLength2 = d.FromKeyTuple("bbox_length2", x => x.DArr.ToList()),
                BboxPhi = d.FromKeyTuple("bbox_phi", x => x.DArr.ToList())
            });
        }

        Samples = list;
    }

    public IEnumerable<string> Errors()
    {
        var errorList = new List<string>();
        if (Ids == null) errorList.Add($"{nameof(Ids)}为空");
        if (ImageDir.IsNullOrWhiteSpace()) errorList.Add("图片路径为空");
        if (Names == null) errorList.Add($"{nameof(Names)}为空");
        if (Samples == null) errorList.Add($"{nameof(Samples)}为空");

        if (Names != null && Ids != null && Ids.Count != Names.Count)
            errorList.Add($"{nameof(Ids)}与{nameof(Names)}数量不一致");

        Ids?.检查重复(errorList, l => $"有重复的Id:{l}");
        Names?.检查重复(errorList, l => $"有重复的Name:{l}");
        Samples?.检查重复(errorList, l => $"有重复的Sample:{l}", new ImageInfoComparer());

        if (Samples != null)
            foreach (var sample in Samples)
                errorList.AddRange(sample.Errors());

        return errorList;
    }

    public HDict ToHDict()
    {
        var dict = new HDict();
        dict.SetDictTuple("class_ids", new HTuple(Ids!.ToArray()));
        dict.SetDictTuple("class_names", new HTuple(Names!.ToArray()));
        dict.SetDictTuple("image_dir", new HTuple(ImageDir));
        var tuple = new HTuple();

        if (Samples != null)
            foreach (var hDict in Samples.Select(sample => sample.ToHDict()))
                tuple.Append(hDict);

        dict.SetDictTuple("samples", tuple);

        return dict;
    }

    public class Sample : IImageInfo, IHalconErrors
    {

        public List<double>? BboxCol { get; set; }

        public List<long>? BboxLabelId { get; set; }

        public List<double>? BboxLength1 { get; set; }

        public List<double>? BboxLength2 { get; set; }

        public List<double>? BboxPhi { get; set; }

        public List<double>? BboxRow { get; set; }

        public IEnumerable<string> Errors()
        {
            var errors = new List<string>();
            if (Id == null) errors.Add($"{nameof(Id)}为空");
            if (FileName.IsNullOrWhiteSpace()) errors.Add($"{nameof(FileName)}为空");
            if (BboxLabelId == null) errors.Add($"{nameof(BboxLabelId)}为空");
            if (BboxRow == null) errors.Add($"{nameof(BboxRow)}为空");
            if (BboxCol == null) errors.Add($"{nameof(BboxCol)}为空");
            if (BboxLength1 == null) errors.Add($"{nameof(BboxLength1)}为空");
            if (BboxLength2 == null) errors.Add($"{nameof(BboxLength2)}为空");
            if (BboxPhi == null) errors.Add($"{nameof(BboxPhi)}为空");

            if (BboxRow != null &&
                BboxLabelId != null &&
                BboxLabelId.Count != BboxRow.Count)
                errors.Add($"{nameof(BboxLabelId)}与{nameof(BboxRow)}数量不一致");

            if (BboxCol != null &&
                BboxLabelId != null &&
                BboxLabelId.Count != BboxCol.Count)
                errors.Add($"{nameof(BboxLabelId)}与{nameof(BboxCol)}数量不一致");

            if (BboxLength1 != null &&
                BboxLabelId != null &&
                BboxLabelId.Count != BboxLength1.Count)
                errors.Add($"{nameof(BboxLabelId)}与{nameof(BboxLength1)}数量不一致");

            if (BboxLength2 != null &&
                BboxLabelId != null &&
                BboxLabelId.Count != BboxLength2.Count)
                errors.Add($"{nameof(BboxLabelId)}与{nameof(BboxLength2)}数量不一致");

            if (BboxPhi != null &&
                BboxLabelId != null &&
                BboxLabelId.Count != BboxPhi.Count)
                errors.Add($"{nameof(BboxLabelId)}与{nameof(BboxPhi)}数量不一致");

            return errors;
        }

        public string? FileName { get; set; }

        public long? Id { get; set; }

        public HDict ToHDict()
        {
            var dict = new HDict();
            dict.SetDictTuple("image_id", new HTuple(Id));
            dict.SetDictTuple("image_file_name", new HTuple(FileName));
            dict.SetDictTuple("bbox_label_id", new HTuple(BboxLabelId?.ToArray()));
            dict.SetDictTuple("bbox_row", new HTuple(BboxRow?.ToArray()));
            dict.SetDictTuple("bbox_col", new HTuple(BboxCol?.ToArray()));
            dict.SetDictTuple("bbox_length1", new HTuple(BboxLength1?.ToArray()));
            dict.SetDictTuple("bbox_length2", new HTuple(BboxLength2?.ToArray()));
            dict.SetDictTuple("bbox_phi", new HTuple(BboxPhi?.ToArray()));

            return dict;
        }

    }

}
