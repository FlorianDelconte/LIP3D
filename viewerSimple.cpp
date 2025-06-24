#include "geometry_utils.h"
#include "image_profiles.h"
#include "customViewer3D.h"

#include <DGtal/base/Common.h>
#include <DGtal/helpers/StdDefs.h>
#include <DGtal/io/readers/MeshReader.h>
#include <DGtal/io/viewers/Viewer3D.h>
#include <DGtal/io/Color.h>
#include <DGtal/shapes/MeshVoxelizer.h>
#include <DGtal/math/linalg/EigenDecomposition.h>
#include "DGtal/io/writers/PGMWriter.h"
#include "DGtal/images/ConstImageAdapter.h"
#include "DGtal/io/writers/VolWriter.h"
#include "DGtal/kernel/BasicPointFunctors.h"

#include <fstream>
#include <QApplication>
#include <QColor>
#include "CLI11.hpp"

using namespace DGtal;
using namespace Z3i;
using namespace std;

// Fonction pour remplir un plan de voxels perpendiculaire à un axe principal dans le domaine spécifié
void fillVoxelPlaneInDomain(Image3D &image, const Z3i::RealPoint &normal, const Z3i::Point &center, unsigned int value) {
    // Calcul de la constante du plan d = n·c
    double d = normal.dot(center);

    // Parcours du domaine de l'image
    for (auto it = image.domain().begin(); it != image.domain().end(); ++it) {
        Z3i::Point pt = *it;
        // Calcul de la distance entre le point et le plan |n·p - d|
        double dist = std::abs(normal[0] * pt[0] + normal[1] * pt[1] + normal[2] * pt[2] - d);

        // Si le point est proche du plan (distance < 0.5), on le marque
        if (dist < 0.5) { 
            image.setValue(pt, value); // Colorer le voxel avec la valeur donnée (par exemple, 255 pour rouge)
        }
    }
}
// Fonction pour ajouter un plan de voxel perpendiculaire a une normal au viewer.
void 
displayPlane(Viewer3D<> &viewer, Image3D::Domain PCLDo, Color c, Z3i::Point offsetCenter, DGtal::Z3i::RealPoint n) {
  //domaine du plan est le meme que celui du volume 3D
  Image3D voxelPlaneImage(PCLDo);
  for (Image3D::Domain::ConstIterator it = voxelPlaneImage.domain().begin(); it != voxelPlaneImage.domain().end(); it++) {
    voxelPlaneImage.setValue(*it, 0); // Initialiser à zéro
  }
  // Remplir le plan de voxels perpendiculaire à la direction principale avec la couleur rouge
  fillVoxelPlaneInDomain(voxelPlaneImage, n, offsetCenter, 255); // 255 pour le rouge
  // Afficher voxelPlaneImage
  viewer << CustomColors3D(c, c); // Couleur rouge pour le plan
  for (auto it = PCLDo.begin(); it != PCLDo.end(); ++it) {
    Z3i::Point pt = *it;
    if (voxelPlaneImage(pt) > 0) { // Si le voxel est sur le plan, l'afficher
      viewer << pt; 
    }
  }
}


void displayProfileImage(CustomViewer3D &viewer, const Image2D &image,
                         const Z3i::RealPoint &normal, const Z3i::Point &center,
                         double width, const Image3D &meshVolImage,
                         const Color &colorIn = Color(0, 0, 0),
                         const Color &colorOut = Color(255, 255, 255)) {
    DGtal::functors::Point2DEmbedderIn3D<DGtal::Z3i::Domain> embedder(meshVolImage.domain(), center, normal, width);

    for (auto it = image.domain().begin(); it != image.domain().end(); ++it) {
        if (image(*it) != 0) {
            Z3i::Point pt3D = embedder(*it);
            viewer << CustomColors3D(colorIn, colorIn);
            viewer << pt3D;
        }else{
            Z3i::Point pt3D = embedder(*it);
            viewer << CustomColors3D(colorOut, colorOut);
            viewer << pt3D;
        }
    }
}


int main(int argc, char **argv)
{
    // Command line parsing
    std::string inputFileName;
    double imageSize = 100.0;
    unsigned char fillValue = 1;
    int maxScan = imageSize;

    CLI::App app{"Simple DGtal Mesh Viewer with Voxelization"};
    app.add_option("-i,--input", inputFileName, "Input OFF mesh file")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("-f,--finesse", imageSize, "Input OFF mesh file");
    CLI11_PARSE(app, argc, argv);

    // Load mesh
    Mesh<RealPoint> mesh;
    MeshReader<RealPoint>::importOFFFile(inputFileName, mesh);

    // Normalize and translate mesh
    auto bbox = mesh.getBoundingBox();
    double diagDist = (bbox.first - bbox.second).norm();
    double meshScale = imageSize / diagDist;
    RealPoint translate = -bbox.first;

    for (auto it = mesh.vertexBegin(), itend = mesh.vertexEnd(); it != itend; ++it) {
        *it += translate;
        *it *= meshScale;
    }
    
    // Voxelization
    Image3D::Domain aDomain(mesh.getBoundingBox().first - Point::diagonal(maxScan),
                            mesh.getBoundingBox().second + Point::diagonal(maxScan));
    DigitalSet mySet(aDomain);
    MeshVoxelizer<DigitalSet, 26> voxelizer;
    voxelizer.voxelize(mySet, mesh, 1.0);
    Image3D meshVolImage(aDomain);
    for(auto p: mySet){
        meshVolImage.setValue(p, fillValue);
    }
    

    auto dir = getMainDirsFromVoxels(meshVolImage); 


    //Image2D profil_sec = compute2D_profile(std::get<1>(dir).getNormalized(), imageSize, meshVolImage, aDomain2D, maxScan);
    //Image2D profil_thir = compute2D_profile(std::get<2>(dir).getNormalized(), imageSize, meshVolImage, aDomain2D, maxScan);
    // Viewer initialization
    QApplication appViewer(argc, argv);
    CustomViewer3D viewer;
    
    

    viewer.show();

    viewer.changeDefaultBGColor(Color(255,255,255, 255));
    // Display mesh
    //viewer << mesh;
    
    viewer << CustomColors3D(Color(50, 50, 50, 255), Color(50, 50, 50, 255));
    for (auto it = aDomain.begin(); it != aDomain.end(); ++it) {
        Z3i::Point pt = *it;
        if (meshVolImage(pt) > 0) {
            viewer << pt;
        }
    }

    Z3i::Point ptL = meshVolImage.domain().lowerBound();
    Z3i::Point ptU = meshVolImage.domain().upperBound();
    Z3i::Point center = (ptL + ptU) / 2;

    
    Image3D::Domain pclDom(mesh.getBoundingBox().first,
                            mesh.getBoundingBox().second);
    /* Affichage du plan principale */
    Z3i::Point offsetCenterPrin = moveCenterAlongDirection(center, -std::get<0>(dir).getNormalized(), -10);
    displayPlane(viewer, pclDom, Color(255, 0, 0, 100), offsetCenterPrin, std::get<0>(dir).getNormalized());
    /* Affichage du plan secondaire */
    //Z3i::Point offsetCenterSec = moveCenterAlongDirection(center, std::get<1>(dir).getNormalized(), 7);
    //displayPlane(viewer, pclDom, Color(0, 255, 0, 100), offsetCenterSec, std::get<1>(dir).getNormalized());
    /* Affichage du plan secondaire */
    //Z3i::Point offsetCenterThir = moveCenterAlongDirection(center, std::get<2>(dir).getNormalized(),7);
    //displayPlane(viewer, pclDom, Color(255, 0, 255, 100), offsetCenterThir, std::get<2>(dir).getNormalized());
    /*Affichage du l'image de profile principale*/
    //compute profile
    
    Image2D::Domain aDomain2D(DGtal::Z2i::Point(0,0),DGtal::Z2i::Point(imageSize-1, imageSize-1));
    maxScan = estimateMaxScanFromDirection(meshVolImage, std::get<0>(dir).getNormalized());
    Image2D profil_main = compute2D_profile(std::get<0>(dir).getNormalized(), imageSize, meshVolImage, aDomain2D, 1);
    Z3i::Point offsetCenterProfile = moveCenterAlongDirection(center, -std::get<0>(dir).getNormalized(), 70);
    displayProfileImage(viewer, profil_main, -std::get<0>(dir).getNormalized(), offsetCenterProfile, imageSize, meshVolImage, Color(0, 0, 0), Color(180, 180, 180, 120));
    
    // Update viewer and run application
    viewer <<  CustomViewer3D::updateDisplay;
    //Save PGM
    //PGMWriter<Image2D>::exportPGM("test_m.pgm", profil_main);
    //PGMWriter<Image2D>::exportPGM("test_s.pgm", profil_sec);
    //PGMWriter<Image2D>::exportPGM("test_t.pgm", profil_thir);
    return appViewer.exec();
}