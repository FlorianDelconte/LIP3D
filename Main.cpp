#include "geometry_utils.h"
#include "image_profiles.h"

#include <DGtal/io/readers/MeshReader.h>
#include <DGtal/shapes/MeshVoxelizer.h>
#include <DGtal/io/writers/PGMWriter.h>
#include <DGtal/io/viewers/Viewer3D.h>
#include <CLI11.hpp>

#include <chrono>
#include <iostream>

using namespace DGtal;
using namespace Z3i;
using namespace std::chrono;

int main(int argc, char **argv) {
    std::string inputFileName, outputPrefix = "output";
    double imageSize = 100.0;
    int maxScan = 50;

    CLI::App app{"Profile image generator"};
    app.add_option("-i,--input", inputFileName, "Input OFF mesh file")->required();
    app.add_option("-o,--output", outputPrefix, "Output image prefix");
    CLI11_PARSE(app, argc, argv);

    auto t0 = high_resolution_clock::now();

    // 1. Lecture du maillage
    Mesh<RealPoint> mesh;
    MeshReader<RealPoint>::importOFFFile(inputFileName, mesh);
    auto t1 = high_resolution_clock::now();
    std::cout << "[1] Mesh loading done in "
              << duration_cast<duration<double>>(t1 - t0).count() << " s\n";

    // 2. Normalisation
    auto bbox = mesh.getBoundingBox();
    double diag = (bbox.first - bbox.second).norm();
    double scale = imageSize / diag;
    RealPoint translate = -bbox.first;

    for (auto it = mesh.vertexBegin(); it != mesh.vertexEnd(); ++it) {
        *it += translate;
        *it *= scale;
    }
    auto t2 = high_resolution_clock::now();
    std::cout << "[2] Normalization done in "
              << duration_cast<duration<double>>(t2 - t1).count() << " s\n";

    // 3. Voxelisation
    Image3D::Domain domain(mesh.getBoundingBox().first - Point::diagonal(maxScan),
                           mesh.getBoundingBox().second + Point::diagonal(maxScan));
    DigitalSet set(domain);
    MeshVoxelizer<DigitalSet, 26> voxelizer;
    voxelizer.voxelize(set, mesh, 1.0);
    
    Image3D volImage(domain);
    for (auto p : set) volImage.setValue(p, 1);
    
    auto t3 = high_resolution_clock::now();
    std::cout << "[3] Voxelization done in "
              << duration_cast<duration<double>>(t3 - t2).count() << " s\n";

    // 4. PCA
    auto dir = getMainDirsFromVoxels(volImage);
    auto t4 = high_resolution_clock::now();
    std::cout << "[4] PCA done in "
              << duration_cast<duration<double>>(t4 - t3).count() << " s\n";

    // 5. Génération des images de profils
    Image2D::Domain domain2D(Z2i::Point(0, 0), Z2i::Point(imageSize - 1, imageSize - 1));
    
    
    auto im0 = compute2D_profile(std::get<0>(dir).getNormalized(), imageSize, volImage, domain2D, maxScan);

    auto im1 = compute2D_profile(std::get<1>(dir).getNormalized(), imageSize, volImage, domain2D, maxScan);

    auto im2 = compute2D_profile(std::get<2>(dir).getNormalized(), imageSize, volImage, domain2D, maxScan);

    auto t5 = high_resolution_clock::now();
    std::cout << "[5] Profile image generation done in "
              << duration_cast<duration<double>>(t5 - t4).count() << " s\n";

    // 6. Sauvegarde des images
    PGMWriter<Image2D>::exportPGM(outputPrefix + "_m.pgm", im0);
    PGMWriter<Image2D>::exportPGM(outputPrefix + "_s.pgm", im1);
    PGMWriter<Image2D>::exportPGM(outputPrefix + "_t.pgm", im2);
    auto t6 = high_resolution_clock::now();
    std::cout << "[6] Saving images done in "
              << duration_cast<duration<double>>(t6 - t5).count() << " s\n";

    std::cout << "Total time: "
              << duration_cast<duration<double>>(t6 - t0).count() << " s\n";

    return 0;
}
