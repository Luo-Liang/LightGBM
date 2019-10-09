/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <LightGBM/application.h>
#include <easy/profiler.h>
#include <iostream>

int main(int argc, char** argv) {
  EASY_PROFILER_ENABLE;
  try {
    LightGBM::Application app(argc, argv);
    app.Run();
    profiler::dumpBlocksToFile("lightgbm.network.prof");
  }
  catch (const std::exception& ex) {
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex.what() << std::endl;
    exit(-1);
  }
  catch (const std::string& ex) {
    std::cerr << "Met Exceptions:" << std::endl;
    std::cerr << ex << std::endl;
    exit(-1);
  }
  catch (...) {
    std::cerr << "Unknown Exceptions" << std::endl;
    exit(-1);
  }
}
