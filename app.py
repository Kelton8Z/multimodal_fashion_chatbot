import os
from pathlib import Path
from argparse import Namespace

import json
import webbrowser

from jina import Flow, Executor, requests
from jina.helper import countdown
import logging
from helper import (
    print_result,     write_html,     download_data,     index_generator,     query_generator,     colored, )
from executors import CLIPEncoder, MyIndexer, MyEvaluator, MyTransformer

cur_dir = os.path.dirname(os.path.abspath(__file__))


def hello_world():
    """
    Description:
        It downloads Amazon fashion dataset and :term:`Indexer<indexes>` 50,000 images.
        The index is stored into 4 *shards*. It randomly samples 128 unseen images as :term:`Queries<Searching>`
        Results are shown in a webpage.


    :param args: Argparse object
    """

    # Path(args.workdir).mkdir(parents=True, exist_ok=True)
    image_files = list(os.walk('./data'))
    '''
    [('./', ['stamps', 't_shirts'], []), 
    ('./stamps', [], ['8.jpg', '9.jpg', '14.jpg', 'stamp1.jpeg']), 
    ('./t_shirts', [], ['shirt1.jpeg', 'shirt3.png', 'shirt2.png'])]
    '''
    stamps = list(map(lambda x: image_files[1][0]+'/'+x, image_files[1][2]))
    t_shirts = list(map(lambda x: image_files[2][0]+'/'+x, image_files[2][2]))
    image_files = stamps + t_shirts
    targets = {
        # "index-labels": {
        #     "label": args.items['product_name'],
        #     "filename": os.path.join(args.workdir, "index-labels"),
        # },
        "index": {
            'data': image_files
            # "image_urls": args.items['large'],
            # "filename": os.path.join(args.workdir, "index-original"),
        },
    }

    # download the data
    # download_data(targets, args.download_proxy)
    #
    # # reduce the network load by using `fp16`, or even `uint8`
    # os.environ["JINA_ARRAY_QUANT"] = "fp16"
    # os.environ["HW_WORKDIR"] = args.workdir

    # now comes the real work CLIPEncoder
    f = Flow(cors=True, restful=True).add(uses=CLIPEncoder, parallel=2).add(uses=MyIndexer).add(uses=MyEvaluator)
    # run it!
    with f:
        input = list(index_generator(num_docs=len(targets["index"]["data"]), target=targets))
        f.post(on='/index', inputs=input, show_progress=True
            #request_size=args.request_size,
        )
        f.protocol = 'http'
        f.port_expose = 8080

        # wait for couple of seconds
        countdown(
            3,
            reason=colored(
                "behold! im going to switch to query mode",
                "cyan",
                attrs=["underline", "bold", "reverse"],
            ),
        )

        """
        f.post(
            '/eval',
            query_generator(
                num_docs=args.num_query, target=targets, with_groundtruth=True
            ),
            shuffle=True,
            on_done=print_result,
            request_size=args.request_size,
            parameters={'top_k': args.top_k},
        )"""

        # write result to html
        # write_html(os.path.join("static/chatbot.html"))

        url_html_path = 'file://' + os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'static/chatbot.html'
            )
        )

        try:
            webbrowser.open(url_html_path, new=2)
        except:
            pass  # intentional pass, browser support isn't cross-platform
        finally:
            logging.info(
                f'You should see a demo page opened in your browser, '
                f'if not, you may open {url_html_path} manually'
            )

        f.block()


if __name__ == "__main__":
    hello_world()
#     with open(
#         "marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson",
#         "r",
#     ) as f:
#         items = f.read().split('\n')
#         selected_fields = ["uniq_id", "product_url",
#                             "product_name",
#                             "large",
#                             "brand",
#                             "sales_price",
#                             "rating",
#                             "sales_rank_in_parent_category",
#                             "sales_rank_in_child_category",
#                             "delivery_type",
#                             "meta_keywords",
#                             "best_seller_tag__y_or_n",
#                             "other_items_customers_buy",
#                             "product_details__k_v_pairs"]
#         filtered_items = []
#         for item in items:
#             try:
#                 item = json.loads(item)
#                 filtered_item = {}
#                 for k,v in item.items():
#                     if k in selected_fields:
#                         if k in ['large', 'other_items_customers_buy']:
#                             filtered_item[k] = v.split('|')
#                         filtered_item[k] = v
#                 filtered_items.append(filtered_item)
#             except json.decoder.JSONDecodeError:
#                 print(item)
#         # large_image_urls = items[0]['large']
#
#         #for filtered_item in filtered_items:
#         args = Namespace(workdir='292303e8-0083-43ed-b0c7-93cec48a1e88', download_proxy=None, items=filtered_items)
#                          #index_data_urls='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
#                          #index_labels_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
# #                  request_size=1024, num_query=128, top_k=50)
