using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Word2vec.Tools;

namespace hackathon201706_tsne
{
    class Program
    {
        static void Main(string[] args)
        {
            Word2VecBinaryReader binfile = new Word2VecBinaryReader();
            Vocabulary v = binfile.Read("f:\\googlenews.bin");

            Console.WriteLine($"Length {v.Words.Length}");
            Console.WriteLine($"Vector Dimensions: {v.VectorDimensionsCount}");

            //var closest = v.Distance("Trump", 10);
            //foreach(var w in closest)
            //{
            //    Console.WriteLine($"{w.Representation.WordOrNull}\t{w.DistanceValue}");
            //}

            //var ro = v.GetRepresentationFor("Obama");
            //var rt = v.GetRepresentationFor("Trump");
            //foreach (var ca in v.Distance(v["Obama"].Substract(v["President"]).Add(v["Money"]), 10))
            //{
            //    Console.WriteLine($"{ca.Representation.WordOrNull}");
            //}


            string[] teststr = new string[2];
            teststr[0] = @"Democrats scrambled to regroup on Wednesday after a disappointing special election defeat in Georgia, with lawmakers, activists and labor leaders speaking out in public and private to demand a more forceful economic message heading into the 2018 elections. Among Democrats in Washington, the setback in Georgia revived or deepened a host of existing grievances about the party, accentuating tensions between moderate lawmakers and liberal activists and prompting some Democrats to question the leadership and political strategy of Nancy Pelosi, the House minority leader. A small group of Democrats who have been critical of Ms. Pelosi in the past again pressed her to step down on Wednesday. And in a private meeting of Democratic lawmakers, Representative Tony Cárdenas of California, Ms. Pelosi’s home state, suggested the party should have a more open conversation about her effect on its political fortunes.";
            teststr[1] = @"democrats scrambled to regroup on wednesday after a disappointing special election defeat in georgia with lawmakers activists and labor leaders speaking out in public and private to demand a more forceful economic message heading into the 2018 elections among democrats in washington the setback in georgia revived or deepened a host of existing grievances about the party accentuating tensions between moderate lawmakers and liberal activists and prompting some democrats to question the leadership and political strategy of nancy pelosi the house minority leader a small group of democrats who have been critical of ms pelosi in the past again pressed her to step down on wednesday and in a private meeting of democratic lawmakers representative tony cárdenas of california ms pelosis home state suggested the party should have a more open conversation about her effect on its political fortunes";
            //teststr[0] = "Obama's proposes 40 million in spending";
            //teststr[1] = "Republicans cut global warming from Obama budget";
            //var teststr = "Obama's new budget proposes 40 million in spending for swag";
            //var teststr2 = "Republicans cut global warming from Obama budget";

            foreach (string str in teststr)
            {
                Console.WriteLine(str);
                Representation sumk = v.GetSummRepresentationOrNullForPhrase(str);
                foreach (var ca in v.Distance(sumk, 10))
                {
                    Console.WriteLine($"{ca.Representation.WordOrNull}");
                }

            }



            //var analogy = v.Analogy("Obama", "budget", "Bush", 10);
            //foreach(var a in analogy)
            //{
            //    Console.WriteLine($"{a.Representation.WordOrNull}");
            //}

            //Word2vec.Tools.Representation r = new Representation()

            //Word2vec.Tools.Vocabulary v = new Vocabulary()
        }
    }
}
