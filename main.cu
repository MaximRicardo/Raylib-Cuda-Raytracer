//This program turns my computer into an inefficient heater

#include <iostream>
#include <memory>
#include <vector>
#include "raylib.h"

#include "OBJ_Loader.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define Screen_Width 1024    //Screen resolution
#define Screen_Height 640

#define Up_Scaled_Width 1280   //Upscaled screen size
#define Up_Scaled_Height 960

#define HFOV (90.f*DEG2RAD)
#define VFOV (65.f*DEG2RAD)

#define MAX_REFLECTIONS 10  //Maximum number of times a ray can reflect off objects

#define RAYS_PER_PIXEL 10    //Number of rays shot through each pixel (More = less noise and more even AA)

#define RAY_SPREAD 1.f        //How far can the rays spread from the center of the pixel (Controls AA)

#define BLENDED_FRAMES 100      //How many frames are averaged to reduce noise


__host__ __device__ float sign(float val) {
    if (val > 0.f) return 1.f;
    else if (val < 0.f) return -1.f;
    return 0.f;
}

__host__ __device__ float RandomVal(unsigned int* state) {

    *state = *state * 747796405 + 2891336453;
    unsigned int result = ((*state >> ((*state >> 28) + 4)) ^ *state) * 277803737;
    result = (result >> 22) ^ result;
    return result / 4294967295.f;

}

__host__ __device__ float RandomValNormalDistr(unsigned int* state) {

    float theta = 2.f * PI * RandomVal(state);
    float rho = std::sqrtf(-2.f * std::logf(RandomVal(state)));
    return rho * std::cosf(theta);

}

Vector3 ObjV3ToV3(objl::Vector3 vec) {
    return { vec.X, vec.Y, vec.Z };
}

__host__ __device__ Vector3 Vec3DAdd(Vector3 a, Vector3 b) {

    return { a.x + b.x,a.y + b.y,a.z + b.z };

}

__host__ __device__ Vector3 Vec3DSub(Vector3 a, Vector3 b) {

    return { a.x - b.x,a.y - b.y,a.z - b.z };

}

__host__ __device__ Vector3 Vec3DMul(Vector3 a, float b) {  //vec * scalar

    return { a.x * b,a.y * b,a.z * b };

}

__host__ __device__ Vector3 Vec3DDiv(Vector3 a, float b) {  //vec / scalar

    return { a.x / b,a.y / b,a.z / b };

}

__host__ __device__ float Vec3DDot(Vector3 a, Vector3 b) {

    return a.x * b.x + a.y * b.y + a.z * b.z;

}

__host__ __device__ Vector3 Vec3DCross(Vector3 a, Vector3 b) {

    return { a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x };

}

__host__ __device__ float Vec3DMagnitude(Vector3 a) {

    return std::sqrtf(Vec3DDot(a, a));

}

__host__ __device__ Vector3 RotateVecAboutVec3D(Vector3 a, Vector3 b, float angle) {

    Vector3 aInB = Vec3DMul(b, Vec3DDot(a, b) / Vec3DDot(b, b));
    Vector3 aInBOrtho = Vec3DSub(a, aInB);

    Vector3 w = Vec3DCross(b, aInBOrtho);

    float x1 = std::cos(angle) / std::sqrtf(Vec3DDot(aInBOrtho, aInBOrtho));
    float x2 = std::sin(angle) / std::sqrtf(Vec3DDot(w, w));

    Vector3 aInBOrthoAngle = Vec3DMul(Vec3DAdd(Vec3DMul(aInBOrtho, x1), Vec3DMul(w, x2)), std::sqrtf(Vec3DDot(aInBOrtho, aInBOrtho)));

    Vector3 rotatedA = Vec3DAdd(aInBOrthoAngle, aInB);

    return rotatedA;

}

__host__ __device__ Color ColorMul(Color a, float b) {
    a.r = fmin((float)a.r * b, 255.f);
    a.g = fmin((float)a.g * b, 255.f);
    a.b = fmin((float)a.b * b, 255.f);

    return a;
}

class Polygon {

public:
    Vector3 points[3];
    Color color;

    Vector3 normal;     //Normal of the polygon's plane
    float d;            //Plane coefficient d

    float smoothness = 0.f;     //from 0 to 1

    float metallic = 0.f;       //from 0 to 1

    float brightness = 0.f;     //from 0 to infinity

    float transparency = 0.f;   //from 0 to 1

    bool reflects = true;

    unsigned int type = 0;       //0 - Triangle, 1 - Sphere

    Vector3 pos;
    float radius;

    void GetTriangleNormal() {
        
        //Get the plane normal
        normal = Vec3DCross(Vec3DSub(points[1], points[0]), Vec3DSub(points[2], points[0]));
        normal = Vec3DDiv(normal, std::sqrtf(Vec3DDot(normal, normal)));    //Normalize the normal

        d = Vec3DDot(normal, points[0]);    //Get the d coefficient

    }

};

__host__ __device__ float RayPlaneIntersect(Vector3 rayStart, Vector3 rayHeading, Vector3 planeNormal, float d) {

    //Determine where the ray intersects the plane
    return (d - Vec3DDot(planeNormal, rayStart)) / Vec3DDot(planeNormal, rayHeading);

}

__host__ __device__ bool PointInTriangle(Polygon pol, Vector3 p) {
    
    //idk but it works
    return (Vec3DDot(Vec3DCross(Vec3DSub(pol.points[1], pol.points[0]), Vec3DSub(p, pol.points[0])), pol.normal) >= 0.f &&
        Vec3DDot(Vec3DCross(Vec3DSub(pol.points[2], pol.points[1]), Vec3DSub(p, pol.points[1])), pol.normal) >= 0.f &&
        Vec3DDot(Vec3DCross(Vec3DSub(pol.points[0], pol.points[2]), Vec3DSub(p, pol.points[2])), pol.normal) >= 0.f);

}

__host__ __device__ Vector3 RandomDirection(unsigned int* state) {

    for (int i = 0; i < 100; i++) {
        Vector3 pointInCube;
        pointInCube.x = RandomValNormalDistr(state) * 2.f - 1.f;
        pointInCube.y = RandomValNormalDistr(state) * 2.f - 1.f;
        pointInCube.z = RandomValNormalDistr(state) * 2.f - 1.f;
        float sqrDist = Vec3DDot(pointInCube, pointInCube);
        return Vec3DDiv(pointInCube, std::sqrtf(sqrDist));
        
    }
    return { 0.f };

}

__device__ Vector3 CastRay(Polygon* pols, int n, Vector3 pos, Vector2 scrPos, Vector3 camPos, unsigned int* randState) {

    //Get the heading vector
    Vector3 heading = Vec3DSub(pos, camPos);
    //Normalize the heading
    heading = Vec3DDiv(heading, std::sqrtf(Vec3DDot(heading, heading)));

    Vector3 rayColor = { 0.f };

    float totalLight = 0.f;

    int lastCollIndex = -1;

    Vector3 lastColor = { 0.f };

    bool lastWasRefract = false;
    
    float colDiv = 1.f;

    int i;
    for (i = 0; i < MAX_REFLECTIONS; i++) {   //Loop n times for a maximum of n reflections

        int collIndex = -1;  //If the collision index is -1, then the ray didn't hit anything
        float dist = 10000000.f;

        for (int j = 0; j < n; j++) {   //Go through every polygon

            //Don't let a ray reflect off the same object twice
            if (j == lastCollIndex) continue;

            //Get t in the equation R(t)=P+tD, where R(t) is on the plane
            if (pols[j].type == 0) {    //Triangle
                float t = RayPlaneIntersect(pos, heading, pols[j].normal, pols[j].d);
                if (t <= 0.f) continue; //Make sure t isn't negative or equal to 0, only planes infront of the ray
                Vector3 p = Vec3DAdd(pos, Vec3DMul(heading, t));    //Convert t to a 3D point
                //If the point is in the triangle, and is the closest hit so far
                if (PointInTriangle(pols[j], p) && t < dist) {
                    dist = t;   //Set this to the closest hit
                    collIndex = j;   //And set this index to the collision index
                }
            }
            else if (pols[j].type == 1) {   //Sphere
                Vector3 offsetPos = Vec3DSub(pos, pols[j].pos);
                float sqrtPartSquared = std::powf(Vec3DDot(offsetPos, heading), 2.f) - (Vec3DDot(offsetPos, offsetPos) - pols[j].radius * pols[j].radius);
                if (sqrtPartSquared < 0.f) continue;
                float t = Vec3DDot(Vec3DMul(offsetPos, -1.f), heading) - std::sqrtf(sqrtPartSquared);
                if (t < dist && t > 0.f) {
                    dist = t;
                    collIndex = j;
                }
            }

        }

        if (collIndex != -1) {   //If it hit something


            Vector3 normal;
            if (pols[collIndex].type == 0) {
                normal = pols[collIndex].normal;
            }
            else if (pols[collIndex].type == 1) {
                Vector3 hitPoint = Vec3DAdd(pos, Vec3DMul(heading, dist));
                normal = Vec3DDiv(Vec3DSub(hitPoint, pols[collIndex].pos), pols[collIndex].radius);
            }

            //Add to the total luminosity that the ray has hit
            totalLight += pols[collIndex].brightness;

            //Get the color of the polygon that is hit
            Vector3 currentCol = { (float)pols[collIndex].color.r, (float)pols[collIndex].color.g, (float)pols[collIndex].color.b };
            
            //Lambert's cosine law
            float lightStrength = std::powf(fabsf(Vec3DDot(normal, heading)), 2.f);
            currentCol = Vec3DMul(currentCol, lightStrength);

            //Only if this is reflected light
            if (i > 0 && !lastWasRefract) {
                //If the light was reflected then tint based on how metallic the material is
                Vector3 lastReflectionColor = Vec3DAdd(currentCol, Vec3DMul(Vec3DSub(lastColor, currentCol), pols[lastCollIndex].metallic));

                //Add the last ray color now that we know the color that is being reflected
                //And also divide by i instead of i+1, because this is for the last ray
                rayColor = Vec3DAdd(rayColor, Vec3DDiv(lastReflectionColor, colDiv / 2.f));
            }
            
            //The current ray color is not added right now because either
            //A) it reflects, and hits something else, which means the program ends up at the if statement above
            //so color is still added eventually
            //B) it reflects, and hits nothing, and if that happens it will still add the color

            //If the polygon doesn't reflect light, then don't reflect
            if (!pols[collIndex].reflects) {
                rayColor = Vec3DAdd(rayColor, Vec3DDiv(lastColor, colDiv));
                break;
            }

            //The darker the color, the less likely the light is to reflect
            //The brighter it is, the more likely the light is to reflect
            //This makes black, for example, not reflect any light.
            //While white will reflect all the light that hits it
            constexpr float maxSquareColorMagnitude = 441.f; //Largest magnitude the color can have
            float reflectVal = RandomVal(randState);    //Random value from 0 to 441
            reflectVal = fmodf(fabsf(reflectVal), maxSquareColorMagnitude+1.f);
            //Get the square magnitude of the polygon's color
            float colorMagnitude = pols[collIndex].color.r * pols[collIndex].color.r + pols[collIndex].color.g * pols[collIndex].color.g + pols[collIndex].color.b * pols[collIndex].color.b;
            colorMagnitude = std::sqrtf(colorMagnitude);
            //If the random number is less than the square magnitude of the color, then don't reflect
            //But do reflect if the ray is supposed to pass through the object
            if (reflectVal > colorMagnitude) {
                break;
            }

            //Decide if the ray will pass through the object, but only if it is transparent
            float transparentVal = RandomVal(randState);
            transparentVal = fmodf(transparentVal, 1.f);

            //Update the position of the ray
            pos = Vec3DAdd(pos, Vec3DMul(heading, dist));

            //Diffuse reflection
            float oldHeadingDotSign = sign(Vec3DDot(heading, normal));
            Vector3 diffHeading = RandomDirection(randState);
            diffHeading = Vec3DMul(diffHeading, -oldHeadingDotSign*sign(Vec3DDot(diffHeading, normal)));

            //If transparentVal is greater than the transparency, then do not let the ray pass through
            if (transparentVal > pols[collIndex].transparency) {
                //Specular reflection, only needed if the ray is reflecting and not refracting
                Vector3 specHeading = Vec3DSub(heading, Vec3DMul(normal, 2.f * Vec3DDot(heading, normal)));

                float linearInterpVal = pols[collIndex].smoothness * (1.f - lightStrength * (1.f - pols[collIndex].metallic));
                //Linearly interpolate between the diffuse and specular reflections depending on smoothness and the angle the ray hit at (unless the object is metallic)
                heading = Vec3DAdd(diffHeading, Vec3DMul(Vec3DSub(specHeading, diffHeading), linearInterpVal));

                lastWasRefract = false;
            }
            else {
                //If transparentVal is less than or equal to the transparency, then let the ray pass through
                //Interpolate based on transparency instead
                //And also use -heading instead of specular heading
                heading = Vec3DAdd(diffHeading, Vec3DMul(Vec3DSub(Vec3DMul(heading, -1.f), diffHeading), pols[collIndex].transparency));
                //And then flip the heading, so it goes through the material
                heading = Vec3DMul(heading, -1.f);

                lastWasRefract = true;
            }

            lastCollIndex = collIndex;

            lastColor = currentCol;

            colDiv *= 2.f - pols[collIndex].smoothness / 2.f;

        }
        else {      //If it didn't hit anything

            if (i > 0)
                rayColor = Vec3DAdd(rayColor, Vec3DDiv(lastColor, colDiv/2.f));

            //Draw black as the last hit color
            rayColor = Vec3DAdd(rayColor, Vec3DDiv({ 0.f }, colDiv));

            //Break out of the loop
            break;

        }

    }

    //Multiply the color by the total brightness
    rayColor = Vec3DMul(rayColor, totalLight);
    //Get the average color over all the hits, taking less into account transparent objects
    rayColor = Vec3DDiv(rayColor, (float)(i + 1));
    //Limit each value to 255
    rayColor.x = fminf(rayColor.x, 255.f);
    rayColor.y = fminf(rayColor.y, 255.f);
    rayColor.z = fminf(rayColor.z, 255.f);

    //Return the final color
    return rayColor;

}

__global__ void RayTrace(unsigned char* colorBuffer, Polygon* pols, int n, Vector3 camPos, Vector2 camAngle, Vector2 normalizedSlope, int frameCount) {

    int indexX = blockIdx.x * blockDim.x + threadIdx.x;
    int indexY = blockIdx.y * blockDim.y + threadIdx.y;

    if (indexX >= Screen_Width || indexY >= Screen_Height) return;

    unsigned int frameCountCopy = frameCount * indexY;

    //Seed for random number generators
    unsigned int randState = ((unsigned int)Screen_Width * (unsigned int)indexY + (unsigned int)indexX) * (frameCount + 1) * 45345235432;

    //indexX and indexY will be the ray's screen position
    Vector2 scrPos = { (float)indexX, (float)indexY };

    //Normalize the screen coordinates
    Vector2 normalizedScrPos;
    normalizedScrPos.x = scrPos.x / Screen_Width * 2.f - 1.f;
    normalizedScrPos.y = scrPos.y / Screen_Height * 2.f - 1.f;

    //Get the would-be location of the pixel if it was in world-space to get the direction of the ray
    //And also to know where the ray would be at the start of the viewing frustum
    //However for anti aliasing, the pixel will be divided into n sub-pixels where n is RAYS_PER_PIXEL, and
    //therefore we need the position of every sub-pixel
    Vector3 pixelPos[RAYS_PER_PIXEL];
    float zNear = 0.01f; //Distance of z near

    for (int i = 0; i < RAYS_PER_PIXEL; i++) {
        //Pick a random location in the pixel for each sub-pixel
        float spreadX = RandomVal(&randState);
        spreadX = fmodf(spreadX, RAY_SPREAD) - (RAY_SPREAD / 2.f);   //Random value from -0.5 to 0.5
        float spreadY = RandomVal(&randState);
        spreadY = fmodf(spreadY, RAY_SPREAD) - (RAY_SPREAD / 2.f);   //Random value from -0.5 to 0.5
        pixelPos[i].z = zNear;
        pixelPos[i].x = HFOV / (PI / 4) * zNear * (normalizedScrPos.x + -normalizedSlope.x * spreadX);
        pixelPos[i].y = VFOV / (PI / 4) * zNear * (normalizedScrPos.y + -normalizedSlope.y * spreadY);
    }

    //Rotate the would-be screen based on the camera rotation
    for (int i = 0; i < RAYS_PER_PIXEL; i++) {
        Vector3 oldPixelPos = pixelPos[i];
        pixelPos[i].y = oldPixelPos.y * std::cos(-camAngle.x) - oldPixelPos.z * std::sin(-camAngle.x);
        pixelPos[i].z = oldPixelPos.y * std::sin(-camAngle.x) + oldPixelPos.z * std::cos(-camAngle.x);

        oldPixelPos = pixelPos[i];
        pixelPos[i].x = oldPixelPos.x * std::cos(-camAngle.y) - oldPixelPos.z * std::sin(-camAngle.y);
        pixelPos[i].z = oldPixelPos.x * std::sin(-camAngle.y) + oldPixelPos.z * std::cos(-camAngle.y);
    }

    //Cast several rays, and then average out the color of each of the rays to get the final pixel color
    //This approach will reduce noise caused by diffuse reflection
    Vector3 totalColor = { 0.f };
    
    for (int i = 0; i < RAYS_PER_PIXEL; i++) {
        //Add the camera pos to the position of the current sub-pixel to get the ray starting pos
        Vector3 rayPos = Vec3DAdd(camPos, pixelPos[i]);
        //Cast a ray
        totalColor = Vec3DAdd(totalColor, CastRay(pols, n, rayPos, scrPos, camPos, &randState));
    }
    //Get the average color
    totalColor = Vec3DDiv(totalColor, RAYS_PER_PIXEL);
    //Convert it to a raylib color
    /*
    Color rayColor;
    rayColor.r = totalColor.x;
    rayColor.g = totalColor.y;
    rayColor.b = totalColor.z;
    rayColor.a = 255;*/

    //Draw it to the buffer
    unsigned int index = (Screen_Width * (Screen_Height - (int)scrPos.y - 1) + (int)scrPos.x) * 4;
    colorBuffer[index] = totalColor.x;
    colorBuffer[index + 1] = totalColor.y;
    colorBuffer[index + 2] = totalColor.z;
    colorBuffer[index + 3] = 255;

}

__global__ void ShiftColorBuffer(unsigned char* colorBuffer) {

    int indexX = blockIdx.x * blockDim.x + threadIdx.x;
    int indexY = blockIdx.y * blockDim.y + threadIdx.y;

    if (indexX >= Screen_Width * 4 || indexY >= Screen_Height) return;

    for (int i = BLENDED_FRAMES-1; i > 0; i--) {
        int index0 = indexX + (Screen_Width * 4) * (indexY + Screen_Height * i);
        int index1 = indexX + (Screen_Width * 4) * (indexY + Screen_Height * (i-1));
        colorBuffer[index0] = colorBuffer[index1];
    }

}

__global__ void BlendColorBuffers(unsigned char* colorBuffer) {

    int indexX = blockIdx.x * blockDim.x + threadIdx.x;
    int indexY = blockIdx.y * blockDim.y + threadIdx.y;

    if (indexX >= Screen_Width * 4 || indexY >= Screen_Height) return;

    float avrg = 0.f;

    for (int i = 0; i < BLENDED_FRAMES; i++) {
        int index = indexX + (Screen_Width * 4) * (indexY + Screen_Height * i);
        avrg += (float)colorBuffer[index];
    }

    avrg /= (float)BLENDED_FRAMES;

    int index = (Screen_Width * 4) * indexY + indexX;
    colorBuffer[index] = (unsigned char)avrg;

}

Vector3 MoveCamForward(Vector3 camPos, Vector2 angle, float moveSpeed, float deltaTime) {

    camPos.x += moveSpeed * deltaTime * sin(angle.y) * cos(angle.x);
    camPos.y += moveSpeed * deltaTime * -sin(angle.x);
    camPos.z += moveSpeed * deltaTime * cos(angle.y) * cos(angle.x);

    return camPos;

}

Vector3 MoveCamBackward(Vector3 camPos, Vector2 angle, float moveSpeed, float deltaTime) {

    camPos.x -= moveSpeed * deltaTime * sin(angle.y) * cos(angle.x);
    camPos.y -= moveSpeed * deltaTime * -sin(angle.x);
    camPos.z -= moveSpeed * deltaTime * cos(angle.y) * cos(angle.x);

    return camPos;

}

Vector3 MoveCamLeft(Vector3 camPos, Vector2 angle, float moveSpeed, float deltaTime) {

    camPos.x -= moveSpeed * deltaTime * sin(angle.y + PI/2.f) * cos(angle.x);
    camPos.y -= moveSpeed * deltaTime * -sin(angle.x);
    camPos.z -= moveSpeed * deltaTime * cos(angle.y + PI / 2.f) * cos(angle.x);

    return camPos;

}

Vector3 MoveCamRight(Vector3 camPos, Vector2 angle, float moveSpeed, float deltaTime) {

    camPos.x += moveSpeed * deltaTime * sin(angle.y + PI / 2.f) * cos(angle.x);
    camPos.y += moveSpeed * deltaTime * -sin(angle.x);
    camPos.z += moveSpeed * deltaTime * cos(angle.y + PI / 2.f) * cos(angle.x);

    return camPos;

}

void GetCubePoints(Vector3* points, Vector3 pos, Vector3 size) {

    Vector3 hSize = Vec3DDiv(size, 2.f);

    points[0] = { pos.x - hSize.x, pos.y - hSize.y, pos.z - hSize.z };
    points[1] = { pos.x - hSize.x, pos.y + hSize.y, pos.z - hSize.z };
    points[2] = { pos.x + hSize.x, pos.y + hSize.y, pos.z - hSize.z };
    points[3] = { pos.x + hSize.x, pos.y - hSize.y, pos.z - hSize.z };
    points[4] = { pos.x - hSize.x, pos.y - hSize.y, pos.z + hSize.z };
    points[5] = { pos.x - hSize.x, pos.y + hSize.y, pos.z + hSize.z };
    points[6] = { pos.x + hSize.x, pos.y + hSize.y, pos.z + hSize.z };
    points[7] = { pos.x + hSize.x, pos.y - hSize.y, pos.z + hSize.z };

}

__global__ void SetAlphaChannel(unsigned char* colorBuffer) {
    for (int i = 3; i < sizeof(unsigned char) * Screen_Width * Screen_Height * 4 * BLENDED_FRAMES; i += 4)
        colorBuffer[i] = 255;
}

int main() {

	int id;
	gpuErrchk(cudaGetDevice(&id));

    InitWindow(Up_Scaled_Width, Up_Scaled_Height, "Window");
    SetTargetFPS(60);

    // Request a texture to render to. The size is the screen size of the raylib example.
    RenderTexture2D renderTexture = LoadRenderTexture(Up_Scaled_Width, Up_Scaled_Height);

    Rectangle source = { 0, (float)-Screen_Height, (float)Screen_Width, (float)-Screen_Height };
    Rectangle sourceBuffer = { 0, (float)Screen_Height, (float)Screen_Width, (float)Screen_Height }; // - Because OpenGL coordinates are inverted
    Rectangle dest = { 0, 0, (float)Up_Scaled_Width, (float)Up_Scaled_Height };

    Image img = GenImageColor(Screen_Width, Screen_Height, BLACK);
    // set the image's format so it is guaranteed to be aligned with our pixel buffer format below
    ImageFormat(&img, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
    Texture tex = LoadTextureFromImage(img);
    

    //Color buffer
    //Regular array as the colorbuffer will not change in size
    std::unique_ptr<unsigned char[]> colorBuffer;
    try {
        colorBuffer = std::make_unique<unsigned char[]>(Screen_Width * Screen_Height * 4);
    }
    catch (std::bad_alloc&) {
        std::cout << "Couldn't allocate colorBuffer\n";
    }
    unsigned char* cudaColorBuffer;
    gpuErrchk(cudaMalloc(&cudaColorBuffer, sizeof(unsigned char) * Screen_Width * Screen_Height * 4 * BLENDED_FRAMES));
    if (cudaColorBuffer == NULL) {
        std::cout << "Couldn't allocate cudaColorBuffer\n";
        return 1;
    }
    gpuErrchk(cudaMemset(cudaColorBuffer, 0, sizeof(unsigned char) * Screen_Width * Screen_Height * 4 * BLENDED_FRAMES));

    SetAlphaChannel<<<1,1>>>(cudaColorBuffer);

    //Get the slope of the normalized coordinates
    Vector2 normalizedPos1;
    normalizedPos1.x = 0.f / Screen_Width * 2.f - 1.f;
    normalizedPos1.y = 0.f / Screen_Height * 2.f - 1.f;

    Vector2 normalizedPos2;
    normalizedPos2.x = 1.f / Screen_Width * 2.f - 1.f;
    normalizedPos2.y = 1.f / Screen_Height * 2.f - 1.f;

    Vector2 normalizedSlope;
    normalizedSlope.x = normalizedPos2.x - normalizedPos1.x;
    normalizedSlope.y = normalizedPos2.y - normalizedPos1.x;

    objl::Loader loader;
    loader.LoadFile("resources/cube.obj");

    //Polygons
    std::vector<Polygon> pols;

    //Create all the polygons
    //You can create whatever polygons you want here
    {
        Polygon pol;
        pol.points[0] = { 1.f,-1.f,2.f };
        pol.points[1] = { 0.f,1.f,2.f };
        pol.points[2] = { -1.f,-1.f,2.f };
        pol.color = RED;
        pol.smoothness = 1.f;
        pol.brightness = 5.f;
        pol.metallic = 1.f;
        pol.GetTriangleNormal();

        Polygon pol2;
        pol2.points[0] = { 1.f,-1.f,3.f };
        pol2.points[1] = { 0.f,1.f,3.f };
        pol2.points[2] = { -1.f,-1.f,3.f };
        pol2.color = BLUE;
        pol2.smoothness = 1.f;
        pol2.brightness = 2.f;
        pol2.metallic = 1.f;
        pol2.GetTriangleNormal();

        Polygon pol3;
        pol3.points[0] = { 4.f,-1.f,1.5f };
        pol3.points[1] = { 3.f,1.f,2.f };
        pol3.points[2] = { 2.f,-1.f,2.5f };
        pol3.color = ORANGE;
        pol3.smoothness = 1.f;
        pol3.brightness = 5.f;
        pol3.metallic = 1.f;
        pol3.GetTriangleNormal();

        Vector3 points[8];
        GetCubePoints(points, { 0.f, 0.f, 0.f }, { 10.f, 3.f, 7.f });

        float boxSmoothness = 1.f;
        float boxMetallic = 1.f;
        Color boxColor = { 57,57,57,255 };

        Polygon pol4;
        pol4.points[0] = points[0];
        pol4.points[1] = points[1];
        pol4.points[2] = points[2];
        pol4.color = boxColor;
        pol4.smoothness = boxSmoothness;
        pol4.metallic = boxMetallic;
        pol4.GetTriangleNormal();

        Polygon pol5;
        pol5.points[0] = points[0];
        pol5.points[1] = points[2];
        pol5.points[2] = points[3];
        pol5.color = boxColor;
        pol5.smoothness = boxSmoothness;
        pol5.metallic = boxMetallic;
        pol5.GetTriangleNormal();

        Polygon pol6;
        pol6.points[0] = points[1];
        pol6.points[1] = points[2];
        pol6.points[2] = points[6];
        pol6.color = boxColor;
        pol6.smoothness = boxSmoothness;
        pol6.metallic = boxMetallic;
        pol6.GetTriangleNormal();

        Polygon pol7;
        pol7.points[0] = points[1];
        pol7.points[1] = points[5];
        pol7.points[2] = points[6];
        pol7.color = boxColor;
        pol7.smoothness = boxSmoothness;
        pol7.metallic = boxMetallic;
        pol7.GetTriangleNormal();

        Polygon pol8;
        pol8.points[0] = points[4];
        pol8.points[1] = points[6];
        pol8.points[2] = points[7];
        pol8.color = boxColor;
        pol8.smoothness = boxSmoothness;
        pol8.metallic = boxMetallic;
        pol8.GetTriangleNormal();

        Polygon pol9;
        pol9.points[0] = points[4];
        pol9.points[1] = points[5];
        pol9.points[2] = points[6];
        pol9.color = boxColor;
        pol9.smoothness = boxSmoothness;
        pol9.metallic = boxMetallic;
        pol9.GetTriangleNormal();

        Polygon pol10;
        pol10.points[0] = points[3];
        pol10.points[1] = points[7];
        pol10.points[2] = points[4];
        pol10.color = boxColor;
        pol10.smoothness = boxSmoothness;
        pol10.metallic = boxMetallic;
        pol10.GetTriangleNormal();

        Polygon pol11;
        pol11.points[0] = points[0];
        pol11.points[1] = points[3];
        pol11.points[2] = points[4];
        pol11.color = boxColor;
        pol11.smoothness = boxSmoothness;
        pol11.metallic = boxMetallic;
        pol11.GetTriangleNormal();

        Polygon pol12;
        pol12.points[0] = points[0];
        pol12.points[1] = points[5];
        pol12.points[2] = points[1];
        pol12.color = boxColor;
        pol12.smoothness = boxSmoothness;
        pol12.metallic = boxMetallic;
        pol12.GetTriangleNormal();

        Polygon pol13;
        pol13.points[0] = points[0];
        pol13.points[1] = points[4];
        pol13.points[2] = points[5];
        pol13.color = boxColor;
        pol13.smoothness = boxSmoothness;
        pol13.metallic = boxMetallic;
        pol13.GetTriangleNormal();

        Polygon pol14;
        pol14.points[0] = points[3];
        pol14.points[1] = points[2];
        pol14.points[2] = points[7];
        pol14.color = boxColor;
        pol14.smoothness = boxSmoothness;
        pol14.metallic = boxMetallic;
        pol14.GetTriangleNormal();

        Polygon pol15;
        pol15.points[0] = points[2];
        pol15.points[1] = points[6];
        pol15.points[2] = points[7];
        pol15.color = boxColor;
        pol15.smoothness = boxSmoothness;
        pol15.metallic = boxMetallic;
        pol15.GetTriangleNormal();

        Polygon pol16;
        pol16.points[0] = { -1.f, 1.4f, -1.f };
        pol16.points[1] = { 0.f, 1.4f, 1.f };
        pol16.points[2] = { 1.f, 1.4f, -1.f };
        pol16.color = WHITE;
        pol16.brightness = 50.f;
        pol16.GetTriangleNormal();

        Polygon sphere;
        sphere.type = 1;
        sphere.pos = { 1.5f, 0.f, -1.5f };
        sphere.radius = 1.f;
        sphere.color = GRAY;
        sphere.smoothness = 1.f;
        sphere.metallic = 1.f;

        pols.push_back(pol);
        pols.push_back(pol2);
        pols.push_back(pol3);
        pols.push_back(pol4);
        pols.push_back(pol5);
        pols.push_back(pol6);
        pols.push_back(pol7);
        pols.push_back(pol8);
        pols.push_back(pol9);
        pols.push_back(pol10);
        pols.push_back(pol11);
        pols.push_back(pol12);
        pols.push_back(pol13);
        pols.push_back(pol14);
        pols.push_back(pol15);
        pols.push_back(pol16);
        pols.push_back(sphere);

        Vector3 meshOffset = { -1.5f, 0.f, -1.5f };
        for (int i = 0; i < loader.LoadedMeshes[0].Vertices.size() / 3; i++) {
            Polygon meshPol;
            for (int j = 0; j < 3; j++) {
                meshPol.points[j] = Vec3DAdd(ObjV3ToV3(loader.LoadedMeshes[0].Vertices[loader.LoadedIndices[j + i * 3]].Position), meshOffset);
            }
            meshPol.brightness = loader.LoadedMeshes[0].MeshMaterial.illum;
            //meshPol.color.r = loader.LoadedMeshes[0].MeshMaterial.Ka.X;
            //meshPol.color.g = loader.LoadedMeshes[0].MeshMaterial.Ka.Y;
            //meshPol.color.b = loader.LoadedMeshes[0].MeshMaterial.Ka.Z;
            //meshPol.color.a = 255;
            meshPol.color = BROWN;
            meshPol.smoothness = 1.f;
            //meshPol.transparency = 0.9f;
            meshPol.metallic = 0.f;

            for (int i = 0; i < 3; i++)
                std::cout << meshPol.points[i].x << ", " << meshPol.points[i].y << ", " << meshPol.points[i].z << '\n';
            std::cout << '\n';

            meshPol.GetTriangleNormal();

            pols.push_back(meshPol);
        }
    }

    //Cuda doesn't support vectors, so an array is used instead
    Polygon* cudaPols;
    gpuErrchk(cudaMalloc(&cudaPols, pols.size() * sizeof(pols[0])));
    if (cudaPols == NULL) {
        std::cout << "Couldn't allocate cudaPols\n";
        return 1;
    }
    gpuErrchk(cudaMemcpy(cudaPols, pols.data(), pols.size() * sizeof(pols[0]), cudaMemcpyHostToDevice));

    Vector2 camAngle = { 0.f, 0.f };
    Vector3 camPos = { 0.f };
    float moveSpeed = 5.f;
    float turnSpeed = 5.f;

    bool blend = false;
    bool paused = false;

    unsigned int frameCount = 0;

    while (!WindowShouldClose()) {

        float deltaTime = GetFrameTime();

        if (IsKeyPressed(KEY_SPACE)) paused = !paused;

        //Camera movement
        if (IsKeyDown(KEY_W)) camPos = MoveCamForward(camPos, camAngle, moveSpeed, deltaTime);
        if (IsKeyDown(KEY_S)) camPos = MoveCamBackward(camPos, camAngle, moveSpeed, deltaTime);
        if (IsKeyDown(KEY_A)) camPos = MoveCamLeft(camPos, camAngle, moveSpeed, deltaTime);
        if (IsKeyDown(KEY_D)) camPos = MoveCamRight(camPos, camAngle, moveSpeed, deltaTime);
        if (IsKeyDown(KEY_Q)) camPos.y -= moveSpeed * deltaTime;
        if (IsKeyDown(KEY_E)) camPos.y += moveSpeed * deltaTime;

        if (IsKeyDown(KEY_R)) camPos.y = 0.f;

        if (IsKeyDown(KEY_LEFT)) camAngle.y -= turnSpeed * deltaTime;
        if (IsKeyDown(KEY_RIGHT)) camAngle.y += turnSpeed * deltaTime;

        camAngle.x = fmodf(camAngle.x, 2.f * PI);
        camAngle.y = fmodf(camAngle.y, 2.f * PI);

        if (camAngle.x < 0.f) camAngle.x += 2.f * PI;
        if (camAngle.y < 0.f) camAngle.y += 2.f * PI;

        if (IsKeyPressed(KEY_T)) {
            //Reset the screen layers
            gpuErrchk(cudaMemset(cudaColorBuffer, 0, sizeof(unsigned char) * Screen_Width * Screen_Height * 4 * BLENDED_FRAMES));
            SetAlphaChannel << <1, 1 >> > (cudaColorBuffer);
            gpuErrchk(cudaDeviceSynchronize());
            frameCount = 0;
        }

        if (IsKeyPressed(KEY_F)) {
            blend = !blend;
        }

        if (!paused) {
            //Cast rays
            dim3 rayTraceBlocks((int)ceilf((float)Screen_Width / 32.f), (int)ceilf((float)Screen_Height / 32.f));
            dim3 rayTraceThreads(32, 32);
            //One thread per pixel
            RayTrace << <rayTraceBlocks, rayTraceThreads >> > (cudaColorBuffer, cudaPols, pols.size(), camPos, camAngle, normalizedSlope, frameCount * blend);

            if (frameCount >= BLENDED_FRAMES && blend) {
                dim3 blendClrBffrBlocks(rayTraceBlocks.x * 10, rayTraceBlocks.y * 10);
                dim3 blendClrBffrThreads(32, 32);
                BlendColorBuffers << <blendClrBffrBlocks, blendClrBffrThreads >> > (cudaColorBuffer);
            }
            //cudaMemcpy implicitly synchronizes the gpu
            //Copy the color buffer from gpu to cpu
            //I can't seem to find a faster way to display the color buffer, because currently the color
            //buffer is copied from video memory, to memory, then back to video memory later to be displayed
            //This causes 3 transfers for literally no reason, while also slowing the program down a bunch
            //Without this, speed should improve immensely, sadly there doesn't seem to be a way to do this
            gpuErrchk(cudaMemcpy(colorBuffer.get(), cudaColorBuffer, sizeof(unsigned char)* Screen_Width* Screen_Height * 4, cudaMemcpyDeviceToHost));

            if (blend) {
                dim3 shiftClrBffrBlocks(rayTraceBlocks.x * 10, rayTraceBlocks.y * 10);
                dim3 shiftClrBffrThreads(32, 32);
                ShiftColorBuffer << <shiftClrBffrBlocks, shiftClrBffrThreads >> > (cudaColorBuffer);

                gpuErrchk(cudaDeviceSynchronize());
            }
        }

        //Draw the color buffer using nested loops (slow but works)
        /*
        BeginTextureMode(renderTexture);
        
        //Draw the color buffer
        for (int y = 0; y < Screen_Height; y++) {
            for (int x = 0; x < Screen_Width; x++) {
                int index = (Screen_Width * y + x)*4;
                DrawPixel(x, y, { colorBuffer[index], colorBuffer[index+1], colorBuffer[index+2], colorBuffer[index+3] });
            }
        }

        EndTextureMode();*/

        BeginDrawing();
        //Draw the color buffer using built in raylib functions (faster but still very slow)
        ClearBackground(BLACK);
        UpdateTexture(tex, colorBuffer.get());
        DrawTexturePro(tex, sourceBuffer, dest, { 0.f, 0.f }, 0.f, WHITE);
        //DrawTexturePro(renderTexture.texture, source, dest, { 0, 0 }, 0.0f, WHITE); Only uncomment if nested loop method is used
        DrawText(TextFormat("Cam Angle Y: %f", camAngle.y * RAD2DEG), 10, 20, 20, WHITE);
        if (blend)
            DrawText("Blend enabled", 10, 50, 20, WHITE);
        else
            DrawText("Blend disabled", 10, 50, 20, WHITE);
        if (paused)
            DrawText("Paused enabled", 10, 80, 20, WHITE);
        else
            DrawText("Paused disabled", 10, 80, 20, WHITE);

        DrawFPS(10, 110);
        EndDrawing();

        ++frameCount;

    }

    UnloadTexture(tex);
    UnloadRenderTexture(renderTexture);

    CloseWindow();

    gpuErrchk(cudaFree(cudaPols));
    
    gpuErrchk(cudaFree(cudaColorBuffer));

    //Make sure all the threads have closed just in case
    gpuErrchk(cudaDeviceSynchronize());

    return 0;

}
